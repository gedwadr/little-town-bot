# Deploys the quantized inference server (port 9003, CPU-only).
#
# Build:
#   docker build -t little-town-bot .
#
# Run:
#   docker run -p 9003:9003 little-town-bot

FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch (much smaller than the CUDA wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir flask && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    pip install --no-cache-dir \
        $(grep -v '^torch\|^torchvision\|^torchaudio\|^flask' requirements.txt | tr '\n' ' ')

COPY src/ src/
COPY nn_rl_all_checkpoints/ensemble/v2/game_940_int8.pt /checkpoints/model_int8.pt

ENV QUANTIZED_CHECKPOINT=/checkpoints/model_int8.pt

EXPOSE 9003

CMD ["python", "-m", "src.frozen_nn_quantize_bot_server"]
