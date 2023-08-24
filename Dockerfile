FROM ubuntu:20.04

WORKDIR /sadtalker

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN --mount=type=cache,target=/root/.cache \
    apt-get update -y \
    && apt-get install -y python3-pip wget ffmpeg \
    && pip3 install --upgrade pip
    
RUN --mount=type=cache,target=/root/.cache \
    pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

COPY ./requirements.txt /sadtalker

# Install requirements
RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt

# Copy the rest of the files

RUN --mount=type=cache,target=/root/.cache \
    mkdir ./checkpoints \
    && wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -O  ./checkpoints/mapping_00109-model.pth.tar \
    && wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -O  ./checkpoints/mapping_00229-model.pth.tar \ 
    && wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -O  ./checkpoints/SadTalker_V0.0.2_256.safetensors \
    && wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors -O  ./checkpoints/SadTalker_V0.0.2_512.safetensors \
    && mkdir -p ./gfpgan/weights \
    && wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth -O ./gfpgan/weights/alignment_WFLW_4HG.pth \
    && wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O ./gfpgan/weights/detection_Resnet50_Final.pth \
    && wget -nc https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O ./gfpgan/weights/GFPGANv1.4.pth \
    && wget -nc https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O ./gfpgan/weights/parsing_parsenet.pth 

COPY . /sadtalker

EXPOSE 8000

# Run the app
CMD uvicorn --host "0.0.0.0" --port "8000" api:app
