FROM vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev libhwloc-dev liburing-dev libturbojpeg0-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir ziglang==0.17.0
ENTRYPOINT ["/bin/bash"]
