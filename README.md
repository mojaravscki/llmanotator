# LLM Anotator
Using LLM to image annotation


Overview
Image annotation has long been a challenging task, especially for domain-specific datasets that require accurate class assignments. However, with advancements in models like Grounding DINO and SAM (Segment Anything Model), the process has become more efficient and accessible. Despite these advancements, assigning the correct classes for specific datasets remains a significant challenge.

This repository aims to bridge the gap by leveraging LLMs to streamline and enhance the annotation process for images.

# HOW TO INSTALL

## 1 - create conda env

```
conda create --name myenv python=3.10
```

## 2- git clone env
```
git clone https://github.com/mojaravscki/llmanotator

```

## 3- download groundingdino_swint_ogc.pth

```
mkdir weights
cd weights
curl -L -o groundingdino_swint_ogc.pth https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
```

## 4- download SAM
```
cd..
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

curl -L -o sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## download groundingdino_swint_ogc.pth

```
pip install -r requirements.txt
```


# How to use

```
python gpt.py \
    --config_file config.txt \
    --reference_images_folder references/ \
    --input_images_folder input/ \
    --output_folder output/ \
    --groundingdino_config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_weights weights/groundingdino_swint_ogc.pth \
    --clip_model_dir clip-vit-base-patch32/ \
    --prompt_file prompt.txt \
    --openai_key sk-OPEN_AI_API_KEY \
    --use_lab \
    --patch_width 150 \
    --patch_height 150 \
    --gpt_model "gpt-4o" \
    --target_objects "olive fruit" \
    --persistent

```
