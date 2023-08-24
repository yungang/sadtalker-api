Thanks to Phạm Gia Linh, [source peoject](https://gitlab.com/phamgialinhlx/sad-talker-api)

# How to run
## Manual Installation
Linux:
1. Installing miniconda, python and git.

2. Creating the env and install the requirements.
``` bash
git clone https://github.com/phamgialinhlx/sad-talker-api.git

cd sad-talker-api 

conda create -n sadtalker python=3.8

conda activate sadtalker

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install ffmpeg

pip install -r requirements.txt
```
3. Create `.env` from [.env.example](./.env.example) file 
4. Download models
```bash
bash scripts/download_models.sh
```
5. Host the server.
``` bash
uvicorn --host "0.0.0.0" --port "8000" api:app
```

## Docker Installation
1. Build `sadtalker` image
```
docker build -t sadtalker .
```
2. Create `.env` from [.env.example](./.env.example) file 
3. Run `sadtalker` container
```
docker run --gpus=all --rm -p 8000:8000 -v ./.env:/sadtalker/.env -d --name sadtalker sadtalker
```
***Note***: Remember to volume mount `.env` file to container

## Test the API
The API  will be hosted on port 8000.
Go to http://127.0.0.1:8000/docs to see API documentation.
``` bash
curl -X POST "http://localhost:8000/generate/" -H "Content-Type:application/json" -d '{"image_link": "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/source_image/happy.png","audio_link": "https://github.com/OpenTalker/SadTalker/raw/main/examples/driven_audio/chinese_poem2.wav"}'
```

