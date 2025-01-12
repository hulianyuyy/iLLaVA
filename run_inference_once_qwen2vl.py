import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "free_video_llm"))
import json
from tqdm import tqdm
import torch
from PIL import Image

from packaging import version
from decord import VideoReader, cpu
import numpy as np
import math
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

import os
import warnings
warnings.filterwarnings("ignore")

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

def is_image_by_extension(file_path):
    _, file_extension = os.path.splitext(file_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    return file_extension.lower() in image_extensions

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')

def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')

def run_inference(args):
    """
    Run inference on one input sample.

    Args:
        args: Command-line arguments.
    """

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)

    if args.illava_llm_k==None:
        pass
    elif isinstance(args.illava_llm_k, str):
        illava_llm_k = args.illava_llm_k.split('-')
        illava_llm_k = [int(layer) for layer in illava_llm_k]
    elif isinstance(args.illava_llm_k, int):
        illava_llm_k = [args.illava_llm_k]

    if args.illava_vit_k==None:
        pass
    if isinstance(args.illava_vit_k, str):
        illava_vit_k = args.illava_vit_k.split('-')
        illava_vit_k = [int(layer) for layer in illava_vit_k]
    elif isinstance(args.illava_vit_k, int):
        illava_vit_k = [args.illava_vit_k]

    if args.enable_illava_llm and args.illava_llm_k==None:
        raise ValueError("illava_llm_k=None when enable_illava_llm")
        
    illava_config = {
                "enable_illava_vit": args.enable_illava_vit,
                "illava_vit_k": illava_vit_k,
                "illava_vit_r": args.illava_vit_r,
                "enable_illava_llm": args.enable_illava_llm,
                "illava_llm_k": illava_llm_k,
                "illava_llm_r": args.illava_llm_r,
            }
    
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2', illava_config=illava_config, 
            )

    # default processer
    processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    input_question = ' '.join(args.question.split('_'))
    if is_image_by_extension(args.input_path):  # single-image case
        # Messages containing multiple images and a text query
        messages = [
            {
                #'role': 'system', 'content': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "role": "user",
                "content": [
                    {"type": "image", "image": ensure_image_url(args.input_path), "min_pixels":args.min_pixels, "max_pixels":args.max_pixels},
                    {"type": "text", "text": input_question},
                ],
            }
        ]
    elif os.path.isdir(args.input_path): # Multi-image case
        # Messages containing multiple images and a text query
        messages = [
            {
                #'role': 'system', 'content': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "role": "user",
                "content": [
                    {"type": "text", "text": input_question},
                ],
            }
        ]

        for x in os.listdir(args.input_path):
            if is_image_by_extension(x):
                messages['content'].append(
                    {"type": "image", "image": ensure_image_url(os.path.join(args.input_path, x)), "min_pixels":args.min_pixels, "max_pixels":args.max_pixels}
                    )

    elif os.path.splitext(args.input_path)[-1] in VIDEO_FORMATS: # video case
        messages = [
            {
                #'role': 'system', 'content': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": ensure_video_url(args.input_path),
                        # "max_pixels": 360 * 420,
                        "nframes": args.max_frames_num,
                    },
                    {"type": "text", "text": input_question},
                ],
            }
        ]    
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"The outputs from model: {output_text}")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Directory to the image file, video file or path to multi-images.", required=True)
    parser.add_argument("--model_path", type=str, help="Directory to the pretrained weights of models.", required=True)
    parser.add_argument("--question", type=str, help="The input question accompanied with the image/video.", required=True, default='describe_the_input')
    parser.add_argument("--max_frames_num", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=16384 * 28 * 28)
    
    # Params for iLLaVA
    parser.add_argument("--enable_illava_vit", type=bool, default=False)
    parser.add_argument("--illava_vit_k", type=str, default=None)   # Input with format like '2-3' denoting layers 2 and 3
    parser.add_argument("--illava_vit_r", type=float, default=0)
    parser.add_argument("--enable_illava_llm", type=bool, default=False)
    parser.add_argument("--illava_llm_k", type=str, default=None)   # Input with format like '2-3' denoting layers 2 and 3
    parser.add_argument("--illava_llm_r", type=float, default=0)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
