# Shopping Sessions

C++20 processes the CSV inputs on parallel shards. PyArrow writes the final daily tables as Snappy Parquet with DATE columns. The Docker image targets the scorer's Linux amd64 Xeon with AVX-512.

## Implementation plan

1. Resolve ID links with union-find and choose the byte-wise smallest alias for each component. Build read-only lookup tables for the worker threads.
2. Memory-map input parts, split them on line boundaries, and scan CSV delimiter and quote masks with AVX-512. Partition candidate events by exact event ID and select the earliest ingest. Keep invalid candidates eligible for deduplication so they suppress later copies.
3. Partition surviving events and valid clicks by canonical person. Sort by timestamp within the person and accumulate session and user-day statistics.
4. Credit the latest qualifying click in a forward pass. Merge overlapping seven-day intervals per person and campaign and count session starts inside each interval. This bounds assist work even for users with many clicks.
5. Stage column buffers on scratch and encode independent groups as Parquet in parallel. Compare both schemas and rows against every fixture and the official dev answer.
6. Time from process startup to exit on generated practice-scale data. Use phase timings to guide further optimization. Confirm the two-second goal on the hidden practice scorer, then measure the 60-million-event final scale separately.

## Build and verify

Install Polars for the comparison tools, then use the starter's Docker harness.

```sh
python3 -m pip install 'polars==2.1.0'
make build
make fixtures
make dev
make run-dev
make check-dev ANSWER=dev-answer
```

`make dev` expects the official `generate` binary on PATH. Download `generator.zip` and `dev-answer.zip` from the Shopping Sessions track page. Use a fresh output directory for a scorer-equivalent run.

To benchmark 20 million events locally, generate data outside the repository and run the image with the scorer's resource settings.

```sh
generate --seed 197 --events 20000000 --clicks 5000000 --output /path/to/practice
time scripts/docker_run.sh /path/to/practice /path/to/practice-out
```

Repeat three times with fresh output directories and take the median. Set `SHOPPING_TIMINGS=1` in the container environment for native phase timings on stderr. The wall clock is the score and includes startup, I/O, and Parquet encoding.

For a native development run on Linux amd64 with AVX-512:

```sh
python3 -m pip install -r requirements.txt
g++ -std=c++20 -O3 -DNDEBUG -pthread -march=x86-64-v4 pipeline/native.cpp -o pipeline/shopping-native
NCPU=8 python3 pipeline/pipeline.py /path/to/data /path/to/out
```

The scorer invokes `run.sh` without arguments. It reads `DATA_DIR` and writes `OUTPUT_DIR/user_daily` and `OUTPUT_DIR/campaign_daily`, using `SCRATCH_DIR` for staging. Input files and the container root can be mounted read-only.

## Measured results

The native run matched the official seed-1 dev answer exactly, with 117,038 user-day rows and 27,400 campaign-day rows. All 36 public edge-case fixtures passed.

On a local AMD EPYC workspace with a two-CPU quota, the official generator's seed-197 dataset had 20 million requested events and 5 million requested clicks. Three end-to-end runs took 9.443 s, 6.889 s, and 7.250 s, for a median of **7.250 s**. The practice-sized output passed the cross-table totals and key uniqueness checks. The hidden eight-vCPU scorer and 60-million-event final have yet to be measured.

Parsing took 3.45 s of the median run, so profile CSV scanning, timestamp conversion, and alias lookup first on scorer-class hardware. Aggregation and staging took 1.72 s. If large person groups dominate that phase, benchmark radix sorting their timestamps against the current comparison sort.
