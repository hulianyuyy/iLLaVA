<h1 align="center">iLLaVA</h1>

<p align="center">
<a href="https://arxiv.org/pdf/2412.06263">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2412.06263-red"></a>


_**iLLaVA** is an efficient method for large vision language models by merging visual tokens. It could achieve about **2×** throughput and **1.7× - 2×** memory reduction with comparable performance through merging redundant visual tokens in some certain layers._

https://github.com/user-attachments/assets/86940bcb-2eb8-4cac-97b3-ac9d1aa6c28f

<div align=center>

<h4> The web demo video</h4>
</div>

<div align=center>
<img width="800" src="./figs/framework.png"/>
<h4> Fig.1: The framework of iLLaVA</h4>
</div>

<div align=center>
<img width="800" src="./figs/effectiveness.png"/>
<h4> Fig.2: The efficiency of iLLaVA </h4>
</div>

<div align=center>
<img width="800" src="./figs/generalizability.jpg"/>
<h4> Fig.3: The generalizability of iLLaVA </h4>
</div>

<div align=center>
<img width="800" src="./figs/visualization.png"/>
<h4> Fig.4: The visualization of iLLaVA </h4>
</div>

*Scheduled Updates🔥*

0. - [x] Setup
1. - [x] Inference and Evaluation
2. - [x] Visualizations
3. - [x] Supporting both image and video benchmarks
4. - [x] Demo
5. - [x] Support Qwen2-VL

## 🧨Setup
```bash
conda create -n illava python=3.10
conda activate illava
bash setup.sh
```

*Notice that you should install `numpy=1.x` instead of `numpy=2.x`*

## 🎈Inference
This repo provides the inference code for iLLaVA, implemented based on [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md) or [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL). 

### For LLaVA-OneVision
We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)  to conduct inference with LLaVA-OneVision. You should manually download the pretrained weight for LLaVA-OneVision (e.g., LLaVA-OneVision 7B) from [here](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov), or conducting the following command to download it:

```
pip install -U huggingface_hub
huggingface-cli download --resume-download lmms-lab/llava-onevision-qwen2-7b-ov --local-dir /path_to_your_dir --local-dir-use-symlinks False --resume-download
```
For users who are unable to visit huggingface (e.g., *China*), you can conduct the following command:
```
pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download lmms-lab/llava-onevision-qwen2-7b-ov --local-dir /path_to_your_dir --local-dir-use-symlinks False --resume-download
```

*We use different settings for image benchmarks and video benchmarks of iLLaVA.* Specifically, we reduce 364 tokens in the image encoder, which are merged in layers [5,6,7,8] for image benchmarks, and in layers [3,4,5,6] for video benchmarks. We reduce about 30% tokens in the language model, which are merged in layer 8 for image benchmarks, and in layer 2 for video benchmarks. We observe there exist more feature redundancy in videos which allow more aggressive token merging.

#### Single-image benchmarks and Multi-image benchmarks
`lmms-eval --model llava_onevision_training_free --model_args pretrained=/path_to_your_checkpoint,conv_template=qwen_1_5,model_name=llava_qwen_training_free,device_map=auto,enable_illava_vit=True,illava_vit_k=5-6-7-8,illava_vit_r=92,enable_illava_llm=True,illava_llm_k=8,illava_llm_r=0.70 --task your_benchmark --batch_size 1 --log_samples --log_samples_suffix llava_onevision_7b --output_path ./logs`
#### Video benchmarks
`lmms-eval --model llava_onevision_training_free --model_args pretrained=/path_to_your_checkpoint,conv_template=qwen_1_5,model_name=llava_qwen_training_free,device_map=auto,max_frames_num=32,enable_illava_vit=True,illava_vit_k=3-4-5-6,illava_vit_r=92,enable_illava_llm=True,illava_llm_k=2,illava_llm_r=0.70 --task your_benchmark  --batch_size 1 --log_samples --log_samples_suffix llava_onevision_7b --output_path ./log`

Replace the `/path_to_your_checkpoint` with the path to your downloaded LLaVA-OneVision pretrained weights. Set the `your_benchmark` as your target benchmark, which can be selected from [supported tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) of lmms-eval.

If you are difficult to visit `https://huggingface.co/` (e.g., in *China*), place `HF_ENDPOINT=https://hf-mirror.com` in the beginning of your command.

The log files are saved in `./logs`.
### For Qwen2-VL
We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to conduct inference with Qwen2-VL. The pretrained weights of Qwen2-VL could be **automatically** downloaded during inference. You can also manually download the pretrained weight for Qwen2-VL (e.g., Qwen2-VL 7B) from [here](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), or conducting the following command to download it:

```
pip install -U huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct --local-dir /path_to_your_dir --local-dir-use-symlinks False --resume-download
```
For users who are unable to visit huggingface (e.g., *China*), you can conduct the following command:
```
pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct --local-dir /path_to_your_dir --local-dir-use-symlinks False --resume-download
```

#### Single-image and video benchmarks
```
cd src/VLMEvalKit
python run.py --data your_benchmark --model Qwen2-VL-7B-Instruct-iLLaVA --verbose
```

Set the `your_benchmark` as your target benchmark. The available benchmarks include: MMMU_DEV_VAL(MMMU benchmark), MME(MME benchmark), MMStar(MMStar benchmark), MMBench_DEV_EN(MMBench benchmark), MMVet(MMVet benchmark), Video-MME_128frame_subs(VideoMME benchmark). Other tasks supported by VLMEvalKit but not tested by iLLaVA can be found in [supported tasks](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY).

The detailed args of iLLaVA for Qwen2-VL are defined in line 339 in [config.py](.src/VLMEvalKit/vlmeval/config.py)

If you are difficult to visit `https://huggingface.co/` (e.g., in *China*), place `HF_ENDPOINT=https://hf-mirror.com` in the beginning of your command.

The output files are saved in `./VLMEvalKit/outputs`.
## ✨Visualization: the token merging process

The visualization of the token merging process is only supported for iLLaVA implemented with LLaVA-OneVision due to code issues.
### Single-image benchmarks and Multi-image benchmarks
`lmms-eval --model llava_onevision_training_free --model_args pretrained=/path_to_your_checkpoint,conv_template=qwen_1_5,model_name=llava_qwen_training_free,device_map=auto,enable_illava_vit=True,illava_vit_k=5-6-7-8,illava_vit_r=92,illava_track_vit_source=True,enable_illava_llm=True,illava_llm_k=8,illava_llm_r=0.70,illava_track_llm_source=True --task your_benchmark --batch_size 1 --log_samples     --log_samples_suffix llava_onevision_7b --output_path ./logs`
### Video benchmarks
`lmms-eval --model llava_onevision_training_free --model_args pretrained=/path_to_your_checkpoint,conv_template=qwen_1_5,model_name=llava_qwen_training_free,device_map=auto,max_frames_num=32,enable_illava_vit=True,illava_vit_k=2-3-4-5-6-7-8,illava_vit_r=80,illava_track_vit_source=True,enable_illava_llm=True,illava_llm_k=2,illava_llm_r=0.50,illava_track_llm_source=True,mm_spatial_pool_stride=1     --task your_benchmark --batch_size 1 --log_samples --log_samples_suffix llava_onevision_7b --output_path ./log`

We here set a more aggressive merging procedure for video benchmarks to show better visualization results. You may modify the hyper-parameters by yourself.

Token merging visualizations for different layers would be stored with a prefix of `attention_map_vit_layer_{remained_token_num}.jpg` and `attention_map_llm_layer_{remained_token_num}.jpg` for vit stages and llm stages in the current dir.

*Note that the visualization for images may not be fully spatially aligned, due to the existence of `image_newline` parameter of LLaVA-Onevision.*

## 🍕Inference with one input
We provide a `.py` file to help users use iLLaVA by specifying one input. The acceptable inputs include a single image, multiple images or a video. 

The parameters you need to specify in the command include:
- `model_path`, which indicates the path to the pretrained model.
- `input_path`, which could be the path to an image file, the directory of multiple images or the path to a video file.
- `question`, which is the question proposed by the user. Different words should be separated by `-` for parsing the command. For example, the default input is `describe_the_input`.

Other parameters may refer to the `.py` file.
### For LLaVA-OneVision
We provide `run_inference_once.py` to conduct inference with one input.
#### Example: inputting a single image
`python run_inference_once.py --enable_illava_vit True --illava_vit_k 5-6-7-8 --illava_vit_r 92  --enable_illava_llm True --illava_llm_k 8 --illava_llm_r 0.70 --model_path /path_to_your_checkpoint --question describe_the_input --input_path /path_to_your_image/xxx.jpg`

#### Example: inputting multiple images
`python run_inference_once.py --enable_illava_vit True --illava_vit_k 5-6-7-8 --illava_vit_r 92  --enable_illava_llm True --illava_llm_k 8 --illava_llm_r 0.70 --model_path /path_to_your_checkpoint --question describe_the_input --input_path /path_to_your_images`

#### Example: inputting a video
`python run_inference_once.py --enable_illava_vit True --illava_vit_k 3-4-5-6 --illava_vit_r 92  --enable_illava_llm True --illava_llm_k 2 --illava_llm_r 0.70 --model_path /path_to_your_checkpoint --question describe_the_input --input_path /path_to_your_video/xxx.mp4`

You could set `--max_frames_num 32` to set different input frames.

#### Visualization
You could add `--illava_track_vit_source True --illava_track_llm_source True` in the command to enable visualization. 

For an image/images, we recommend using the following command:

`python run_inference_once.py --enable_illava_vit True --illava_vit_k 2-3-4-5-6-7-8 --illava_vit_r 80  --enable_illava_llm True --illava_llm_k 8 --illava_llm_r 0.50 --model_path /path_to_your_checkpoint --question describe_the_input  --input_path /path_to_your_image.jpg `

For videos, we recommend using `mm_spatial_pool_stride=1` and larger merging steps to enable better visualization. 

`python run_inference_once.py --enable_illava_vit True --illava_vit_k 2-3-4-5-6-7-8 --illava_vit_r 80  --enable_illava_llm True --illava_llm_k 2 --illava_llm_r 0.50 --model_path /path_to_your_checkpoint --question describe_the_input --mm_spatial_pool_stride 1  --input_path /path_to_your_video/xxx.mp4 `

### For Qwen2-VL
We provide `run_inference_once_qwen2vl.py` to conduct inference with one input.
#### Example: inputting a single image
`python run_inference_once_qwen2vl.py --enable_illava_vit True --illava_vit_k 12-13-14-15 --illava_vit_r 0.16  --enable_illava_llm True --illava_llm_k 6-7-8 --illava_llm_r 0.90 --model_path /path_to_your_checkpoint --question describe_the_input --input_path /path_to_your_image/xxx.jpg`

#### Example: inputting multiple images
`python run_inference_once_qwen2vl.py --enable_illava_vit True --illava_vit_k 12-13-14-15 --illava_vit_r 0.16  --enable_illava_llm True --illava_llm_k 6-7-8 --illava_llm_r 0.90 --model_path /path_to_your_checkpoint --question describe_the_input --input_path /path_to_your_images`

#### Example: inputting a video
`python run_inference_once_qwen2vl.py --enable_illava_vit True --illava_vit_k 12-13-14-15 --illava_vit_r 0.16  --enable_illava_llm True --illava_llm_k 6-7-8 --illava_llm_r 0.90 --model_path /path_to_your_checkpoint --question describe_the_input --input_path /path_to_your_video/xxx.mp4`

You could set `--max_frames_num 32` to set different input frames for input videos.
## 🎄Demo 
We provide a offline demo to help users deploy iLLaVA on their local machines. It supports inputting a single image, multiple images or a video, and the get the outputs from iLLaVA. 

### For LLaVA-OneVision
The command is shown as follows:

`python demo.py --enable_illava_vit True --illava_vit_k 5-6-7-8 --illava_vit_r 92  --enable_illava_llm True --illava_llm_k 8 --illava_llm_r 0.70 --model_path /path_to_your_checkpoint` 

### For Qwen2-VL
The command is shown as follows:

`python demo_qwen2vl.py  --enable_illava_vit True --illava_vit_k 12-13-14-15 --illava_vit_r 0.16  --enable_illava_llm True --illava_llm_k 6-7-8 --illava_llm_r 0.90 --model_path /path_to_your_checkpoint` 


Below is the visualization for our demo.

**Upload an image, multiple images or a video** and **enter a prompt** to get the outputs from iLLaVA

<div align=center>
<img width="1000" src="./figs/demo.jpg"/>
<h4> Fig.5: The visualization of our demo </h4>
</div>

## 🎫Model hyper-parameters
Besides the original paramters of LLaVA-Onevision, we introduce several new paramters:

- `enable_illava_vit[bool]`, whether enables using iLLaVA in the ViT stage. `Default: False`.
- `illava_vit_k[str]`, the layers to merge tokens in the ViT stage. For example, `2-3-4-5` indicates layers [2,3,4,5]. `Default: None`.
- `illava_vit_r[int]`, the number of tokens to be merged in each layer of the ViT stage. The overall number of merged tokens are the multiplication of `illava_vit_k` and `illava_vit_r`. `Default: 0`.
- `enable_illava_llm[bool]`, whether enables using iLLaVA in the LLM stage. `Default: False`.
- `illava_llm_k[str]`, the layers to merge tokens in the LLM stage. For example, `2` indicates layers [2]. `Default: None`.
- `illava_llm_r[int]`, the number of tokens to be merged in each layer of the LLM stage. The overall number of merged tokens are the multiplication of `illava_llm_k` and `illava_llm_r`. `Default: 0`.
- `illava_llm_image_token_start_index[int]`, the starting index of image tokens in the language model, which is predefined according to the system prompts. `Default: 14`. *Note: This is only applicable for LLaVA-OneVision*
- `illava_track_vit_source[bool]`, whether performing visualization for the token merging process in the ViT stage. `Default: False`. *Note: This is only applicable for LLaVA-OneVision*
- `illava_track_llm_source[bool]`, whether performing visualization for the token merging process in the ViT stage. `Default: False`. *Note: This is only applicable for LLaVA-OneVision*

You can set the corresponding parameters in the `model_args` of the command like we provide in the `inference` section.

## 🛒Model inplementation 
### For LLaVA-OneVision
We mainly modify the following files to conduct different functions:
- [llava_qwen_training_free.py](./src/LLaVA-OneVision/llava/model/language_model/llava_qwen_training_free.py), which preprocess the inputs before feeding into the LVLM.
- [siglip_encoder.py](./src/LLaVA-OneVision/llava/model/multimodal_encoder/siglip_encoder.py), which implements the ViT.
- [builder.py](./src/LLaVA-OneVision/llava/model/builder.py), which defines the class `llava_onevision_training_free`
- [modeling_qwen2.py](./src/transformers-4.46.0/src/transformers/models/qwen2/modeling_qwen2.py), which implements the forward pass of LLM.
- [llava_onevision_training_free.py](./src/lmms_eval-0.2.4/lmms_eval/models/llava_onevision_training_free.py), which defines the model `llava_onevision_training_free` of model initialization and generation process.
### For Qwen2-VL
We mainly modify the following files to conduct different functions:
- [model.py](./src/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py), which defines the Qwen2 model and builds prompts.
- [modeling_qwen2_vl.py](./src/transformers-4.46.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py), which implements the forward pass of image encoder and LLM.
## 🎁Acknowledgements

Thanks to [FastV](https://github.com/pkunlp-icler/FastV), [FreeVideoLLM](https://github.com/contrastive/FreeVideoLLM) for their open-source code.