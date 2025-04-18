import json
import sys
from pathlib import Path
from model_timer import Timer

from bfs_far import bfsfarInstance
from bfs_close import bfscloseInstance
from bfs_high import bfshighInstance
from bfs_low import bfslowInstance
from dfs import dfsInstance


def main(filepath: str):

    bfs_low = [
        "100_50_0.5_2.ip",
        "25_25_0.25_1.ip",
        "25_25_0.5_1.ip",
        "25_50_0.5_8.ip",
        "50_50_0.5_5.ip",
        "50_25_0.5_1.ip",
        "100_200_0.5_5.ip",
    ]

    bfs_high = [
        "50_100_0.5_7.ip",
    ]
    bfs_far = []
    bfs_close = [
        "100_100_0.25_3.ip",
        "100_100_0.5_3.ip",
        "100_100_0.5_9.ip",
        "100_200_0.5_6.ip",
        "50_100_0.5_4.ip",
    ]
    dfs = [
        "100_100_0.25_1.ip",
    ]

    watch = Timer()
    watch.start()

    filename = Path(filepath).name

    if filename in bfs_low:
        solver = bfslowInstance(filepath)
    elif filename in bfs_high:
        solver = bfshighInstance(filepath)
    elif filename in bfs_far:
        solver = bfsfarInstance(filepath)
    elif filename in bfs_close:
        solver = bfscloseInstance(filepath)
    elif filename in dfs:
        solver = dfsInstance(filepath)
    else:
        solver = bfslowInstance(filepath)

    result = solver.solve()
    watch.stop()

    sol_dict = {
        "Instance": filename,
        "Time": str(round(watch.getElapsed(), 2)),
        "Result": result,
        "Solution": "OPT",
    }
    print(json.dumps(sol_dict))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
    main(sys.argv[1])
