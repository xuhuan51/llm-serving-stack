import datetime
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29599")

    torch.cuda.set_device(0)
    print(
        f"rank={rank} init MASTER={master_addr}:{master_port} "
        f"visible={os.environ.get('NVIDIA_VISIBLE_DEVICES')}",
        flush=True,
    )
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )

    x = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    print(f"rank={rank} all_reduce_result={x.item()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
