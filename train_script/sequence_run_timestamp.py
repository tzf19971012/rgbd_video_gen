"""Give all ROMA nodes the timestamp chosen by node 0, before training starts.

Uses a CPU TCPStore without initializing a device/process group. Its short-lived
server uses a separate port from accelerate. Only the timestamp is written to
stdout; diagnostics go to stderr.
"""

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def resolve_timestamp(master_addr, master_port, num_nodes, node_rank,
                      timezone="Asia/Shanghai", timestamp="", timeout=600):
    if num_nodes < 1 or not 0 <= node_rank < num_nodes or timeout <= 0:
        raise ValueError("Invalid node count, node rank or timeout")
    if timestamp:
        parsed = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        if parsed.strftime("%Y%m%d_%H%M%S") != timestamp:
            raise ValueError("RUN_TIMESTAMP must have format YYYYMMDD_HHMMSS")
    if num_nodes == 1:
        return timestamp or datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d_%H%M%S")

    # 每个节点只启动一个辅助进程。主节点提供服务，其他节点等待其上线。
    # 不依赖共享文件中可能残留的旧时间戳，也不启动 NPU 或 HCCL。
    from torch.distributed import TCPStore
    store = TCPStore(master_addr, master_port, num_nodes, node_rank == 0,
                     timedelta(seconds=timeout), use_libuv=False)
    if node_rank == 0:
        chosen = timestamp or datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d_%H%M%S")
        store.set("sequence_timestamp", chosen)
    chosen = store.get("sequence_timestamp").decode("ascii")
    if timestamp and timestamp != chosen:
        raise ValueError("RUN_TIMESTAMP differs between ROMA nodes")

    if node_rank == 0:
        # 等每个节点读到时间戳，再退出并释放监听端口。
        store.wait([f"timestamp_received_{rank}" for rank in range(1, num_nodes)])
    else:
        store.set(f"timestamp_received_{node_rank}", "ready")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    print(resolve_timestamp(args.master_addr, args.master_port, args.num_nodes, args.node_rank,
                            args.timezone, args.timestamp, args.timeout), flush=True)


if __name__ == "__main__":
    main()
