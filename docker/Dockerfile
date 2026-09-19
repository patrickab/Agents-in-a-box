# Keep the image minimal. Mount language runtimes and tools from the profile.
FROM debian:trixie-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]