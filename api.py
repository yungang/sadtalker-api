from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from time import  strftime
import torch
import shutil
import sys
from src.utils.init_path import init_path
import requests
import os
import boto3
from botocore.exceptions import ClientError

from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
import logging

from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY= os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY= os.getenv("AWS_SECRET_KEY")
AWS_S3_REGION= os.getenv("AWS_S3_REGION")
AWS_S3_BUCKET_NAME= os.getenv("AWS_S3_BUCKET_NAME")

def download_file(image_url, save_dir):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception if the request was not successful
        with open(save_dir + "/" + image_url.split('/')[-1], "wb") as f:
            f.write(response.content)
        print(image_url.split('/')[-1], " downloaded and saved successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")

def upload_file_aws(file_name, object_name):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """


    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    region_name=AWS_S3_REGION)
    
    extra_args = {
        'ContentType': 'video/mp4',
        'ACL': 'public-read',  # Add ACL information here
    }

    try:
        response = s3_client.upload_file(file_name, AWS_S3_BUCKET_NAME, object_name, ExtraArgs=extra_args)
        print(f'File "{object_name}" successfully uploaded to S3 bucket "{AWS_S3_BUCKET_NAME}" with object key "{object_name}"')
    except ClientError as e:
        logging.error(e)
        return False
    return True

app = FastAPI()

class Item(BaseModel):
    image_link: str
    audio_link: str
    s3_object_path: str = 'uploads/avatar/'
# https://s3.us-west-1.amazonaws.com/dev.talktent/uploads/audio/d07VqU6UPC.mp3
# In-memory list to store the items (for demonstration purposes)

# curl -X POST "http://localhost:8000/generate /" -H "Content-Type: application/json" -d '{
#   "image_link": "https://zmp3-photo-fbcrawler.zadn.vn/avatars/3/a/6/d/3a6de9f068f10fcee2c50cdf9772ebaa.jpg",
#   "audio_link": "https://s3.us-west-1.amazonaws.com/dev.talktent/uploads/audio/d07VqU6UPC.mp3"
# }'

# Define a POST endpoint to create new items
@app.post("/generate/")
async def sadtalker_create(item: Item):

    # PIC_PATH = "/mnt/work/Code/SadTalker/examples/source_image/art_0.png"
    RESULT_DIR = "./results"
    save_dir = os.path.join(RESULT_DIR, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    download_file(item.image_link, save_dir)
    download_file(item.audio_link, save_dir)
    PIC_PATH = os.path.join(save_dir, item.image_link.split('/')[-1])
    # AUDIO_PATH = "/mnt/work/Code/SadTalker/examples/driven_audio/chinese_poem1.wav"
    AUDIO_PATH = os.path.join(save_dir, item.audio_link.split('/')[-1])
    POSE_STYLE = 0
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    BATCH_SIZE = 2
    INPUT_YAW_LIST = None
    INPUT_PITCH_LIST = None
    INPUT_ROLL_LIST = None
    REF_EYEBLINK = None
    REF_POSE = None
    CHECKPOINT_DIR = "./checkpoints"
    OLD_VERSION = False
    PREPROCESS = "full"
    EXPRESSION_SCALE = 1.0
    STILL = True
    SIZE = 256
    BACKGROUND_ENHANCER = None
    ENHANCER = None
    FACE3DVIS = False
    VERBOSE = False

    current_root_path = './' #os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(current_root_path, 'src/config'), 256, OLD_VERSION, PREPROCESS)
    
        #init model
    preprocess_model = CropAndExtract(sadtalker_paths, DEVICE)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  DEVICE)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, DEVICE)

    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(PIC_PATH, first_frame_dir, PREPROCESS,\
                                                                             source_image_flag=True, pic_size=SIZE)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if REF_EYEBLINK is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(REF_EYEBLINK)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(REF_EYEBLINK, ref_eyeblink_frame_dir, PREPROCESS, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if REF_POSE is not None:
        if REF_POSE == REF_EYEBLINK: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(REF_POSE)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(REF_POSE, ref_pose_frame_dir, PREPROCESS, source_image_flag=False)
    else:
        ref_pose_coeff_path=None
    
    #audio2ceoff
    batch = get_data(first_coeff_path, AUDIO_PATH, DEVICE, ref_eyeblink_coeff_path, still=STILL)
    coeff_path = audio_to_coeff.generate(batch, save_dir, POSE_STYLE, ref_pose_coeff_path)
    
    opt = {
        "net_recon": 'resnet50',
        "init_path": None,
        "use_last_fc": False,
        "bfm_folder": "./checkpoints/BFM_Fitting/",
        "bfm_model": "BFM_model_front.mat",
        "focal": 1015.0,
        "center": 112.0,
        "camera_d": 10.0,
        "z_near": 5.0,
        "z_far": 15.0,
    }

    # 3dface render
    if FACE3DVIS:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(opt, DEVICE, first_coeff_path, coeff_path, AUDIO_PATH, os.path.join(save_dir, '3dface.mp4'))


    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, AUDIO_PATH, 
                                BATCH_SIZE, INPUT_YAW_LIST, INPUT_PITCH_LIST, INPUT_ROLL_LIST,
                                expression_scale=EXPRESSION_SCALE, still_mode=STILL, preprocess=PREPROCESS, size=SIZE)
    
    result = animate_from_coeff.generate(data, save_dir, PIC_PATH, crop_info, \
                                enhancer=ENHANCER, background_enhancer=BACKGROUND_ENHANCER, preprocess=PREPROCESS, img_size=SIZE)
    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    if not VERBOSE:
        shutil.rmtree(save_dir)

    file_path = save_dir + '.mp4'
    print(file_path)
    print(os.path.exists(file_path))
    if item.s3_object_path[-1] != '/':
        item.s3_object_path += '/'
    object_name = item.s3_object_path + os.path.basename(file_path)
    upload_file_aws(file_path, object_name)
    s3_url = f'https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{object_name}'
    os.remove(file_path)
    return s3_url
