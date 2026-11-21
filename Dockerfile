# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS build
WORKDIR /build
COPY pipeline/native.cpp .
RUN apt-get update && apt-get install -y --no-install-recommends g++ && \
    g++ -std=c++20 -O3 -DNDEBUG -pthread -march=x86-64-v4 -static-libstdc++ -static-libgcc native.cpp -o shopping-native && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.12-slim-bookworm
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=secret,id=pip_ca \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi && \
    pip install --no-cache-dir -r requirements.txt
COPY pipeline/pipeline.py /app/pipeline/
COPY --from=build /build/shopping-native /app/pipeline/
COPY --chmod=755 run.sh /app/run.sh
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
CMD ["/app/run.sh"]
