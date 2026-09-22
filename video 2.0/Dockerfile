FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    PYTHONPATH=/app/OmniParserLocal

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN git clone --depth 1 https://github.com/OminousIndustries/OmniParserLocal.git

WORKDIR /app/OmniParserLocal

COPY patch_omniparser.py /tmp/patch_omniparser.py
RUN cp /tmp/patch_omniparser.py patch_omniparser.py && python3 patch_omniparser.py

RUN pip3 install --upgrade pip && \

    pip3 install paddlepaddle==2.6.2 paddleocr==2.7.3 && \
    pip3 install -r requirements.txt && \
    pip3 install huggingface_hub opencv-python-headless

RUN rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence && \
    for f in icon_detect/train_args.yaml icon_detect/model.pt icon_detect/model.yaml \
             icon_caption/config.json icon_caption/generation_config.json icon_caption/model.safetensors; do \
        hf download microsoft/OmniParser-v2.0 "$f" --local-dir weights ; \
    done && \
    mv weights/icon_caption weights/icon_caption_florence

COPY process_video.py /app/OmniParserLocal/process_video.py

RUN mkdir -p /data/input /data/output

ENTRYPOINT ["python3", "/app/OmniParserLocal/process_video.py"]
CMD ["--help"]