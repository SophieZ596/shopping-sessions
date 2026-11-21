"""Run the native pipeline and write its column buffers as Parquet."""
from concurrent.futures import ThreadPoolExecutor
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

USER_SCHEMA = (
    ("user_id", "string"), ("day", "date32"), ("sessions", "int64"),
    ("events", "int64"), ("revenue_cents", "int64"), ("top_category", "nullable_string"),
)
CAMPAIGN_SCHEMA = (
    ("campaign_id", "string"), ("channel", "string"), ("day", "date32"),
    ("sessions", "int64"), ("purchasing_sessions", "int64"),
    ("revenue_cents", "int64"), ("assisted_sessions", "int64"),
)


def read_batch(path, schema, pa):
    with path.open("rb") as file:
        data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    (rows,) = struct.unpack_from("<Q", data)
    position = 8

    def take(size):
        nonlocal position
        start = position
        position += size
        return pa.py_buffer(memoryview(data)[start:position])

    columns = []
    for _, kind in schema:
        if kind in ("string", "nullable_string"):
            valid = take((rows + 7) // 8) if kind == "nullable_string" else None
            (length,) = struct.unpack_from("<i", data, position + 4 * rows)
            buffers = [valid, take(4 * (rows + 1)), take(length)]
            dtype = pa.string()
        else:
            dtype = getattr(pa, kind)()
            buffers = [None, take(rows * (4 if kind == "date32" else 8))]
        columns.append(pa.Array.from_buffers(dtype, rows, buffers))
    return pa.RecordBatch.from_arrays(columns, names=[name for name, _ in schema])


def write_table(pa, pq, paths, schema, output):
    batches = [read_batch(path, schema, pa) for path in paths]
    table = pa.Table.from_batches(batches)
    pq.write_table(table, output, compression="snappy",
                   write_statistics=False, use_dictionary=False)


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_DIR", "/data"))
    output = Path(sys.argv[2] if len(sys.argv) > 2 else os.getenv("OUTPUT_DIR", "/output"))
    workers = os.getenv("NCPU", "8")
    with (data / "manifest.json").open() as manifest_file:
        manifest = json.load(manifest_file)
    binary = Path(__file__).with_name("shopping-native")
    with tempfile.TemporaryDirectory(dir=os.getenv("SCRATCH_DIR")) as directory:
        staging = Path(directory)
        with subprocess.Popen([binary, data, staging, manifest["window_start"],
                               manifest["window_end"], workers]) as process:
            # Overlap Arrow startup with the native pass.
            import pyarrow as pa
            import pyarrow.parquet as pq

            status = process.wait()
            if status:
                raise SystemExit(status)
        user_dir = output / "user_daily"
        campaign_dir = output / "campaign_daily"
        for folder in (user_dir, campaign_dir):
            folder.mkdir(parents=True, exist_ok=True)
        user_paths = sorted(staging.glob("users-*"))
        # Group native shards to bound the number of output files.
        with ThreadPoolExecutor(max_workers=min(8, max(1, int(workers)))) as writer:
            tasks = [writer.submit(write_table, pa, pq, [staging / "campaigns"],
                                   CAMPAIGN_SCHEMA, campaign_dir / "part-0.parquet")]
            for start in range(0, len(user_paths), 8):
                tasks.append(writer.submit(write_table, pa, pq, user_paths[start:start + 8],
                                           USER_SCHEMA, user_dir / f"part-{start // 8}.parquet"))
            for task in tasks:
                task.result()
        for group in range((len(user_paths) + 7) // 8, 8):
            (user_dir / f"part-{group}.parquet").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
