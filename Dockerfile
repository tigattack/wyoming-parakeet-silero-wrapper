FROM python:3.13 AS builder

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
      cmake build-essential

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --target=/install -r requirements.txt

FROM python:3.13-slim

COPY --from=builder /install /usr/local/lib/python3.13/site-packages
COPY . .

VOLUME ["/root/.cache/huggingface"] # Model cache

ENTRYPOINT ["python3", "wyoming_vad_asr_server.py"]
