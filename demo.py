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

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from lmms_eval.models.model_utils.load_video import read_video_pyav

from packaging import version
from decord import VideoReader, cpu
import numpy as np
from llava.conversation import SeparatorStyle, conv_templates
from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
from llava.mm_utils import KeywordsStoppingCriteria
import math
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
os.environ['GRADIO_TEMP_DIR'] = 'gradio_temp'
import gradio as gr
import os
import warnings
warnings.filterwarnings("ignore")

def is_image_by_extension(file_path):
    _, file_extension = os.path.splitext(file_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    return file_extension.lower() in image_extensions

def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

def pad_sequence(tokenizer, input_ids, batch_first, padding_value):
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

def run_inference(inputs, input_question):
    """
    Run inference on one input sample.

    Args:
        args: Command-line arguments.
    """
    print(inputs)
    if isinstance(inputs, Image.Image):  # Single-image case
        image_list = []
        image_list.append(inputs)

        raw_frames = []
        image_tensor = process_images(image_list, image_processor, config)
        if type(image_tensor) is list:
            image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda')
        raw_frames.append((image_tensor[0].permute(0,2,3,1)*127+128).int().cpu())  #  [num_frames, width, height, channels], e.g, [2, 384, 384, 3]

        placeholder_count = len(image_list) if isinstance(image_list, list) else 1
        task_type = 'image'
    elif isinstance(inputs, list):  # Multi-image case
        image_list = []
        for x in inputs:
            image_list.append(Image.open(x))

        raw_frames = []
        image_tensor = process_images(image_list, image_processor, config)
        if type(image_tensor) is list:
            image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda')

        if type(image_tensor) is list:
            image_tensor = torch.stack([_image[0] for _image in image_tensor])  # [num_frames, width, height, channels], e.g, [8, 384, 384, 3]
        else:
            image_tensor = image_tensor[:, 0] # extract the images of resized scale with 384*384
        raw_frames.append((image_tensor*127+128).permute(0,2,3,1).int().cpu())  #  [num_frames, width, height, channels], e.g, [2, 384, 384, 3] 

        placeholder_count = len(image_list) if isinstance(image_list, list) else 1
        task_type = 'image'
    elif os.path.splitext(inputs)[-1] in VIDEO_FORMATS: # Video case
        image_tensor = []
        raw_frames = []

        try:
            if args.video_decode_backend == "decord":
                frames = load_video(inputs, args.max_frames_num)  # (frames, height, width, channels)
            elif args.video_decode_backend == "pyav":
                frames = read_video_pyav(inputs, num_frm=args.max_frames_num)
            frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
            image_tensor.append(frames)
            raw_frames.append((frames*127+128).permute(0,2,3,1).int().cpu())  #  [num_frames, width, height, channels], e.g, [8, 384, 384, 3]
        except Exception as e:
            raise ValueError(f"Error {e} in loading video")

        placeholder_count = 1
        task_type = 'video'

    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
    image_tokens = " ".join(image_tokens)
    question = image_tokens + "\n" + input_question
    
    question_input = []
    conv = conv_templates['qwen_1_5'].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    question_input.append(prompt_question)
    
    input_ids_list = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = pad_sequence(tokenizer, input_ids_list, batch_first=True, padding_value=pad_token_ids).cuda()
    attention_masks = input_ids.ne(pad_token_ids).cuda()

    if task_type == 'image':
        gen_kwargs = {'do_sample': False, 'max_new_tokens': 1024, 'temperature': 0, 'top_p': None, 'num_beams': 1, "image_sizes": [image_list[idx].size for idx in range(len(image_list))]}
    elif task_type == 'video':
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # gen_kwargs = {'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0, 'num_beams': 1, 'do_sample': False, "modalities": ["video"], "stopping_criteria": [stopping_criteria]}
        gen_kwargs = {'do_sample': False, 'max_new_tokens': 1024, 'temperature': 0, 'top_p': None, 'num_beams': 1, 'modalities': ['video'], "stopping_criteria": [stopping_criteria]}
    # These steps are not in LLaVA's original code, but are necessary for generation to work
    # TODO: attention to this major generation step...
    try:
        with torch.inference_mode():
            if use_illava:
                output_ids = model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=True, illava_config=illava_config, raw_frames=raw_frames, **gen_kwargs)
            else:
                output_ids = model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=True, **gen_kwargs)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        raise e
    return outputs


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Directory to the pretrained weights of models.", required=True)
    parser.add_argument("--max_frames_num", type=int, default=32)
    parser.add_argument("--video_decode_backend", type=str, default='decord')
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    
    # Params for iLLaVA
    parser.add_argument("--enable_illava_vit", type=bool, default=False)
    parser.add_argument("--illava_vit_k", type=str, default=None)   # Input with format like '2-3' denoting layers 2 and 3
    parser.add_argument("--illava_vit_r", type=int, default=0)
    parser.add_argument("--enable_illava_llm", type=bool, default=False)
    parser.add_argument("--illava_llm_k", type=str, default=None)   # Input with format like '2-3' denoting layers 2 and 3
    parser.add_argument("--illava_llm_r", type=float, default=0)
    parser.add_argument("--illava_llm_image_token_start_index", type=int, default=14)
    parser.add_argument("--illava_track_vit_source", type=bool, default=False)
    parser.add_argument("--illava_track_llm_source", type=bool, default=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    disable_torch_init()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)

    mm_spatial_pool_stride = args.mm_spatial_pool_stride
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
                "illava_llm_image_token_start_index": args.illava_llm_image_token_start_index,
                "illava_track_vit_source": True if args.illava_track_llm_source and args.enable_illava_vit else args.illava_track_vit_source,
                "illava_track_llm_source": args.illava_track_llm_source,
            }
    if args.enable_illava_vit != True and args.enable_illava_llm != True:
        use_illava = False
    else:
        use_illava = True
    
    llava_model_args = {
        "multimodal": True,
    }
    llava_model_args["attn_implementation"] = best_fit_attn_implementation
    
    model_name = 'llava_qwen_training_free' if use_illava else 'llava_qwen'

    overwrite_config = {}
    overwrite_config["mm_spatial_pool_stride"] = mm_spatial_pool_stride
    overwrite_config["mm_spatial_pool_mode"] = 'bilinear'

    llava_model_args["overwrite_config"] = overwrite_config
    # Try to load the model with the multimodal argument
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map='cuda', **llava_model_args)
    model.eval()
    config = model.config

    def identity(x, y):
        return x

    with gr.Blocks(title='iLLaVA') as demo: 
        gr.Markdown("<center><font size=5>iLLaVA</center></font>")
        gr.Markdown("**Upload an image, multiple images or a video** and **enter a prompt** to get the outputs from iLLaVA.")
        with gr.Tab('Image'):
            with gr.Row():
                with gr.Column(scale=1):
                    Image_input = gr.Image(type="pil",label="Upload an image file")
                    Input_question = gr.Textbox(label="Prompt", placeholder="Describe the input in detail.", lines=2)
                    image_button = gr.Button("Run")  
                with gr.Column(scale=1):
                    image_output = gr.Textbox(label="Output")
        with gr.Tab('Multi-Images'):
            with gr.Row():
                with gr.Column(scale=1):
                    multiple_image_show = gr.Gallery(label="Show the input images", height=200)
                    Multi_image_input = gr.UploadButton(label="Click to upload multiple images", file_types = ['.png','.jpg','.jpeg', '.bmp'], file_count = "multiple")
                    Input_question = gr.Textbox(label="Prompt", placeholder="Describe the input in detail.", lines=2)
                    multiple_image_button = gr.Button("Run")  
                with gr.Column(scale=1):
                    multiple_image_output = gr.Textbox(label="Output")
        with gr.Tab('Video'):
            with gr.Row():
                with gr.Column(scale=1):
                    Video_input = gr.Video(sources=["upload"], label="Upload a video file")
                    Input_question = gr.Textbox(label="Prompt", placeholder="Describe the input in detail.", lines=2)
                    video_button = gr.Button("Run")  
                with gr.Column(scale=1):
                    video_output = gr.Textbox(label="Output")
        image_button.click(run_inference, inputs=[Image_input, Input_question], outputs=image_output)  
        multiple_image_button.click(identity, inputs=[Multi_image_input, Input_question], outputs=multiple_image_show)
        multiple_image_button.click(run_inference, inputs=[Multi_image_input, Input_question], outputs=multiple_image_output)
        video_button.click(run_inference, inputs=[Video_input, Input_question], outputs=video_output)
        
    demo.launch(share=True,server_name="0.0.0.0", server_port=7862)
