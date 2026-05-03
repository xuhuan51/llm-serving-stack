"""
PD Disaggregated Prefill Proxy

Routes OpenAI-compatible chat completion requests through a prefill/decode split:
  1. Tokenize the chat messages
  2. POST /inference/v1/generate to prefill instance (tokens-only, triggers KV transfer)
  3. POST /inference/v1/generate to decode instance with kv_transfer_params
  4. Stream decode response back to client as OpenAI SSE

Measures and logs per-request TTFT and TPOT.
"""

import asyncio
import json
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

PREFILL_URL     = "http://127.0.0.1:8010"
DECODE_URL      = "http://127.0.0.1:8011"
MODEL_NAME      = "qwen-pd"
MODEL_PATH      = "/home/liuguangli/models/Qwen2.5-7B-Instruct"
PROXY_PORT      = 8012

# ZMQ addresses that each vLLM instance binds (kv_port + world_rank=0)
PREFILL_ZMQ_IP   = "192.168.1.101"
PREFILL_ZMQ_PORT = 14579
DECODE_ZMQ_IP    = "192.168.1.101"
DECODE_ZMQ_PORT  = 14580

app = FastAPI()
tokenizer: AutoTokenizer = None


@app.on_event("startup")
async def load_tokenizer():
    global tokenizer
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Tokenizer ready.")


def messages_to_token_ids(messages: list[dict], max_tokens: int) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(text, add_special_tokens=False)


async def call_prefill(
    client: httpx.AsyncClient,
    token_ids: list[int],
    request_id: str,
    model: str,
) -> dict:
    """Send to prefill instance, wait for KV transfer to complete."""
    payload = {
        "request_id": request_id,
        "token_ids": token_ids,
        "sampling_params": {
            "max_tokens": 1,        # just need 1 token to trigger KV transfer
            "temperature": 0.0,
        },
        "model": model,
        "stream": False,
    }
    resp = await client.post(
        f"{PREFILL_URL}/inference/v1/generate",
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


async def stream_decode(
    client: httpx.AsyncClient,
    token_ids: list[int],
    first_token_ids: list[int],
    kv_transfer_params: dict | None,
    request_id: str,
    model: str,
    max_tokens: int,
    temperature: float,
):
    """Send to decode instance and stream SSE chunks back."""
    all_input_ids = token_ids + (first_token_ids or [])
    payload = {
        "request_id": request_id,
        "token_ids": all_input_ids,
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        "model": model,
        "stream": True,
        "kv_transfer_params": kv_transfer_params,
    }

    first_token_time = None
    start = time.perf_counter()
    chunk_count = 0
    all_token_ids = []

    async with client.stream(
        "POST",
        f"{DECODE_URL}/inference/v1/generate",
        json=payload,
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        async for raw_line in response.aiter_lines():
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data_str = raw_line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now

            tids = []
            for choice in chunk.get("choices", []):
                tids.extend(choice.get("token_ids") or [])

            if tids:
                chunk_count += 1
                all_token_ids.extend(tids)
                text = tokenizer.decode(tids, skip_special_tokens=True)
                sse_chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(sse_chunk)}\n\n"

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else elapsed
    tpot = (elapsed - ttft) / (chunk_count - 1) if chunk_count > 1 else 0.0
    full_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    print(f"[decode] id={request_id[:8]} tokens={len(all_token_ids)} "
          f"TTFT={ttft*1000:.1f}ms TPOT={tpot*1000:.1f}ms E2E={elapsed*1000:.1f}ms")

    done_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 200)
    temperature = body.get("temperature", 0.0)
    model = body.get("model", MODEL_NAME)
    # Embed peer ZMQ addresses so P2pNcclConnector.parse_request_id can find them.
    # Prefill reads: ___decode_addr_IP:PORT
    # Decode reads:  ___prefill_addr_IP:PORT___
    base_id = str(uuid.uuid4())
    request_id = (
        f"{base_id}"
        f"___prefill_addr_{PREFILL_ZMQ_IP}:{PREFILL_ZMQ_PORT}___"
        f"decode_addr_{DECODE_ZMQ_IP}:{DECODE_ZMQ_PORT}"
    )
    stream = body.get("stream", True)

    token_ids = messages_to_token_ids(messages, max_tokens)

    t_prefill_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        prefill_resp = await call_prefill(client, token_ids, request_id, model)

    t_prefill_end = time.perf_counter()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000
    kv_transfer_params = prefill_resp.get("kv_transfer_params")
    first_token_ids = []
    for choice in prefill_resp.get("choices", []):
        first_token_ids.extend(choice.get("token_ids") or [])

    print(f"[prefill] id={request_id[:8]} prompt_tokens={len(token_ids)} "
          f"prefill_time={prefill_ms:.1f}ms kv_params={kv_transfer_params is not None}")

    async def generate():
        async with httpx.AsyncClient() as client:
            async for chunk in stream_decode(
                client, token_ids, first_token_ids,
                kv_transfer_params, request_id, model,
                max_tokens, temperature,
            ):
                yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="warning")
