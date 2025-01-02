# LLM Anotator
Using LLM to image annotation


Overview
Image annotation has long been a challenging task, especially for domain-specific datasets that require accurate class assignments. However, with advancements in models like Grounding DINO and SAM (Segment Anything Model), the process has become more efficient and accessible. Despite these advancements, assigning the correct classes for specific datasets remains a significant challenge.

This repository aims to bridge the gap by leveraging LLMs to streamline and enhance the annotation process for images.



python gpt.py \
    --config_file /config.txt \
    --reference_images_folder /references \
    --input_images_folder /input \
    --output_folder /output \
    --groundingdino_config /GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_weights /weights/groundingdino_swint_ogc.pth \
    --clip_model_dir /clip-vit-base-patch32 \
    --prompt_file /path/to/prompt.txt \
    --openai_key <OPEN_API_KEY> \
    --use_lab \
    --patch_width 100 \
    --patch_height 100 \
    --gpt_model "gpt-4o" \
    --target_objects "olive fruit"
