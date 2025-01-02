import os
import time
import base64
import logging
import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from transformers import CLIPModel, CLIPProcessor
from groundingdino.util.inference import load_model, load_image, predict
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from scipy.ndimage import center_of_mass
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
import io
import argparse
import shutil
import csv

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ripeness_assessment.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

##############################################################################
# Utility Functions
##############################################################################
def get_mime_type(image_path: str) -> str:
    """Determine the MIME type based on the image file extension."""
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.bmp': 'image/bmp',
        '.gif': 'image/gif',
    }
    ext = os.path.splitext(image_path)[1].lower()
    return mime_types.get(ext, 'application/octet-stream')

def convert_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an image from RGB to LAB color space."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

def calculate_center_of_mass(image: np.ndarray) -> tuple:
    """Calculate the center of mass of a binary mask or intensity image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized = gray / 255.0  # Normalize intensity
    center = center_of_mass(normalized)
    return int(center[1]), int(center[0])  # Return (x, y)

def extract_center_patch(image: np.ndarray, center: tuple, patch_size=(100, 100)) -> np.ndarray:
    """
    Extract a patch of size patch_size centered at the given center coordinates.
    """
    cx, cy = center
    half_w, half_h = patch_size[0] // 2, patch_size[1] // 2

    x_start, x_end = max(cx - half_w, 0), min(cx + half_w, image.shape[1])
    y_start, y_end = max(cy - half_h, 0), min(cy + half_h, image.shape[0])

    patch = image[y_start:y_end, x_start:x_end]

    if patch.size == 0:
        logging.error(f"[ERROR] Empty patch extracted at center: {center}")
        raise ValueError("Empty patch extracted. Check center coordinates.")

    return patch

def extract_and_save_patch(image: np.ndarray, center: tuple, save_path: str, patch_name: str, patch_size=(100, 100)):
    """Extract a patch centered on the center of mass, save it to disk, and return the patch."""
    patch = extract_center_patch(image, center, patch_size)
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
    save_file = os.path.join(save_path, f"{patch_name}.png")
    cv2.imwrite(save_file, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))  # Save patch as PNG (convert RGB to BGR for OpenCV)
    logging.info(f"[INFO] Patch saved at: {save_file}")
    return patch

##############################################################################
# CLIP Functions
##############################################################################
def resize_image(image: Image.Image, max_size=(800, 800)) -> Image.Image:
    """Resize image to the maximum dimensions while maintaining aspect ratio."""
    image.thumbnail(max_size, Image.ANTIALIAS)
    return image

def encode_image(image_path: str) -> str:
    """Resize and encode the image to a base64 string with appropriate MIME type."""
    mime_type = get_mime_type(image_path)
    
    with Image.open(image_path).convert("RGB") as img:
        img = resize_image(img)
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:{mime_type};base64,{encoded_str}"

def get_image_embedding(image: Image.Image, clip_model: CLIPModel, processor: CLIPProcessor, device: torch.device, use_lab: bool = False) -> np.ndarray:
    """Get normalized CLIP embedding for a PIL image."""
    if use_lab:
        np_image = np.array(image)
        np_image_lab = convert_to_lab(np_image)
        image = Image.fromarray(np_image_lab)

    # Determine if autocast is supported on the device
    if device.type in ['cuda', 'mps']:
        autocast_type = device.type
        with torch.amp.autocast(autocast_type):
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
    else:
        # For CPU, autocast is not necessary
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.squeeze(0).cpu().numpy()
    return embedding

##############################################################################
# GroundingDINO Loader
##############################################################################
def load_groundingdino_model(config_path: str, weights_path: str, device: torch.device = None) -> torch.nn.Module:
    """Load the GroundingDINO model."""
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logging.info(f"[load_groundingdino_model] Using device: {device}")

    model = load_model(config_path, weights_path)
    model.to(device)  # Move model to device
    return model

##############################################################################
# CLIP Loader
##############################################################################
def load_clip_model(clip_model_dir: str, device: torch.device = None) -> tuple:
    """Load the local CLIP model."""
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logging.info(f"[load_clip_model] Using device: {device}")

    clip_model = CLIPModel.from_pretrained(clip_model_dir)
    clip_model.to(device)  # Move model to device
    processor = CLIPProcessor.from_pretrained(clip_model_dir)
    return clip_model, processor, device

##############################################################################
# Vector Store Functions
##############################################################################
def build_reference_vectorstore(reference_images: list, clip_model: CLIPModel, processor: CLIPProcessor, device: torch.device, use_lab: bool = False, patch_save_dir: str = "patches/vector_db", patch_size=(100, 100)) -> FAISS:
    """
    Build a FAISS vector store from reference images with augmentations and center-of-mass patches.
    """
    docs = []
    embeddings_list = []

    def augment_image(image: Image.Image):
        """Apply augmentations: flip, rotate, adjust saturation."""
        augmented_images = [image]

        # Flip horizontally
        augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # Rotate 90° clockwise and counter-clockwise
        augmented_images.append(image.rotate(-90, expand=True))
        augmented_images.append(image.rotate(90, expand=True))

        # Adjust saturation (-25% to +25%)
        enhancer = ImageEnhance.Color(image)
        for saturation_factor in [0.75, 1.25]:  # -25% and +25% adjustments
            augmented_images.append(enhancer.enhance(saturation_factor))

        return augmented_images

    os.makedirs(patch_save_dir, exist_ok=True)

    for i, (img_path, label) in enumerate(reference_images):
        if not os.path.exists(img_path):
            logging.warning(f"[WARNING] File not found: {img_path} — skipping.")
            continue

        logging.info(f"\n[build_reference_vectorstore] Loading reference #{i}: {img_path}")
        pil_img = Image.open(img_path).convert("RGB")
        np_image = np.array(pil_img)

        # Calculate center of mass
        center = calculate_center_of_mass(np_image)

        # Extract patch
        patch = extract_center_patch(np_image, center, patch_size=patch_size)
        patch_pil = Image.fromarray(patch)

        # Save original patch
        original_patch_save_path = os.path.join(patch_save_dir, f"reference_{i}_original.png")
        patch_pil.save(original_patch_save_path)
        logging.info(f"[INFO] Original patch saved at: {original_patch_save_path}")

        # Generate embedding for the original patch
        emb = get_image_embedding(patch_pil, clip_model, processor, device, use_lab=use_lab)
        emb = emb.astype(np.float32).tolist()
        doc = Document(page_content=label, metadata={"label": label})
        docs.append(doc)
        embeddings_list.append(emb)

        # Apply augmentations
        augmented_images = augment_image(patch_pil)

        for aug_idx, aug_img in enumerate(augmented_images):
            # Save each augmented patch
            aug_patch_save_path = os.path.join(patch_save_dir, f"reference_{i}_aug_{aug_idx}.png")
            aug_img.save(aug_patch_save_path)
            logging.info(f"[INFO] Augmented patch saved at: {aug_patch_save_path}")

            # Generate embedding for the augmented patch
            emb = get_image_embedding(aug_img, clip_model, processor, device, use_lab=use_lab)
            emb = emb.astype(np.float32).tolist()

            # Create a document for the vector store
            doc = Document(page_content=f"{label} (Augmentation {aug_idx})", metadata={"label": label})
            docs.append(doc)
            embeddings_list.append(emb)

    if not docs:
        raise ValueError("[ERROR] No valid reference images found!")

    # Minimal Embeddings object for precomputed embeddings
    class NumpyArrayEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return []
        def embed_query(self, text: str) -> list[float]:
            return []

    embedding_function = NumpyArrayEmbeddings()
    text_embedding_tuples = [(doc.page_content, emb) for doc, emb in zip(docs, embeddings_list)]
    all_metadatas = [doc.metadata for doc in docs]

    logging.info(f"\n[build_reference_vectorstore] Building FAISS index from {len(text_embedding_tuples)} items.")
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_tuples,
        embedding=embedding_function,
        metadatas=all_metadatas,
    )
    return vectorstore

##############################################################################
# Orchestrator Functions
##############################################################################
def orchestrate_annotation(chat: ChatOpenAI, vector_db_label: str, target_objects: list, image_description: str = None) -> dict:
    """
    Determines whether to annotate the detected object based on vector DB label and ChatGPT's analysis.

    Parameters:
        chat (ChatOpenAI): Initialized LangChain ChatOpenAI instance.
        vector_db_label (str): Label obtained from the vector database.
        target_objects (list): List of target object names (e.g., ["olive fruit", "apple"]).
        image_description (str, optional): Description of the image for ChatGPT analysis.

    Returns:
        dict: Decision and additional information if applicable.
    """
    # Step 1: Ask ChatGPT if the object matches any of the target objects
    prompt_is_target = (
        f"Based on the following label from the vector database: '{vector_db_label}', "
        f"does this object match any of the following target objects: {', '.join(target_objects)}? "
        "Answer with the matched object name or 'No'."
    )
    human_message = HumanMessage(content=prompt_is_target)
    
    try:
        response_is_target = chat([human_message]).content.strip()
    except Exception as e:
        logging.error(f"[orchestrate_annotation] ChatGPT API call failed: {e}")
        return {"annotate": False, "reason": "ChatGPT API error"}

    # Check if the response matches any target object
    matched_object = None
    for obj in target_objects:
        if response_is_target.lower() == obj.lower():
            matched_object = obj
            break

    if matched_object:
        logging.info(f"[orchestrate_annotation] Matched Object: {matched_object}")

        # Step 2: Ask ChatGPT for specific assessment based on the matched object
        if image_description:
            prompt_assessment = (
                f"Given that the object is a '{matched_object}' with the following description: '{image_description}', "
                f"please assess its ripeness based on the Jaén Scale. Provide a class between 0 to 7."
            )
        else:
            prompt_assessment = (
                f"Given that the object is a '{matched_object}', please assess its ripeness based on the Jaén Scale. "
                "Provide a class between 0 to 7."
            )
        human_message_assessment = HumanMessage(content=prompt_assessment)
        
        try:
            response_assessment = chat([human_message_assessment]).content.strip()
            return {"annotate": True, "matched_object": matched_object, "assessment": response_assessment}
        except Exception as e:
            logging.error(f"[orchestrate_annotation] ChatGPT assessment failed: {e}")
            return {"annotate": True, "matched_object": matched_object, "assessment": "Assessment unavailable"}

    else:
        # Step 3: Decide not to annotate
        return {"annotate": False, "reason": "Not the target object"}

##############################################################################
# CLASSIFY SEGMENTED OBJECTS
##############################################################################
def classify_segmented_objects(
    image_rgb: np.ndarray,
    boxes: list,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    vectorstore: FAISS,
    chat: ChatOpenAI = None,
    top_k: int = 1,
    use_lab: bool = False,
    cropped_save_dir: str = "cropped",
    patch_size: tuple = (100, 100),
    label_to_id_map: dict = {},
    target_objects: list = []
) -> list:
    """Classify each segmented object by matching patches against the reference vectorstore."""
    results = []
    for j, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox
        center_global = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        
        # Extract patch for vector database
        patch = extract_center_patch(image_rgb, center_global, patch_size=patch_size)
        patch_pil = Image.fromarray(patch)
        
        # Generate embedding
        emb = get_image_embedding(patch_pil, clip_model, processor, device, use_lab=use_lab)
        emb = emb.astype(np.float32).reshape(-1)  # Flatten the embedding
        
        # Perform FAISS similarity search
        docs = vectorstore.similarity_search_by_vector(emb, k=top_k)
        
        matches = []
        for k, doc in enumerate(docs):
            label = doc.metadata.get("label", "Unknown")
            matches.append({"label": label, "distance": "N/A"})  # Distance not directly available
            logging.info(f"Box #{j} - Match {k+1}: {label}")
        
        # Extract the entire cropped object for GPT
        cropped_object = image_rgb[y_min:y_max, x_min:x_max]
        cropped_object_pil = Image.fromarray(cropped_object)
        
        # Save the cropped object
        os.makedirs(cropped_save_dir, exist_ok=True)
        cropped_save_path = os.path.join(cropped_save_dir, f"detected_object_{j}.png")
        cropped_object_pil.save(cropped_save_path)
        logging.info(f"[INFO] Detected object image saved at: {cropped_save_path}")
        
        # Initialize image description (you might want to automate this based on actual data)
        image_description = "A high-resolution image of the detected object."

        orchestrator_decision = {"annotate": False, "reason": "No annotation required"}

        # Get orchestrator's decision
        if chat:
            vector_db_label = matches[0]["label"] if matches else "Unknown"
            logging.info(f"[classify_segmented_objects] Vector DB Label: {vector_db_label}")
            orchestrator_decision = orchestrate_annotation(chat, vector_db_label, target_objects=target_objects, image_description=image_description)
            
            if orchestrator_decision.get("annotate"):
                ripeness = orchestrator_decision.get("assessment", "Assessment unavailable")
                matched_object = orchestrator_decision.get("matched_object", "Unknown")
                logging.info(f"[classify_segmented_objects] Matched Object: {matched_object}, Ripeness: {ripeness}")
            else:
                ripeness = orchestrator_decision.get("reason", "No annotation required")
                logging.info(f"[classify_segmented_objects] Annotation skipped: {ripeness}")
        else:
            vector_db_label = matches[0]["label"] if matches else "Unknown"
            ripeness = "N/A"
            logging.info(f"[classify_segmented_objects] OpenAI API not initialized. Skipping annotation.")
        
        # Map label to class ID
        class_id = label_to_id_map.get(vector_db_label, len(label_to_id_map))  # Default to next ID if not found
        
        results.append({
            "bbox": bbox,
            "matches": matches,
            "ripeness": ripeness,
            "class_id": class_id,
            "annotate": orchestrator_decision.get("annotate", False) if chat else False
        })
    return results

##############################################################################
# Annotation Module
##############################################################################
def annotate_image(image_rgb: np.ndarray, results: list, output_image_path: str, labels_dir: str, W_orig: int, H_orig: int):
    """
    Annotate the image with bounding boxes and labels based on results.

    Parameters:
        image_rgb (np.ndarray): Original image in RGB.
        results (list): List containing annotation decisions and details.
        output_image_path (str): Path to save the annotated image.
        labels_dir (str): Directory to save YOLO label files.
        W_orig (int): Original image width.
        H_orig (int): Original image height.
    """
    annotated_frame = image_rgb.copy()
    label_contents = []
    
    for result in results:
        bbox = result["bbox"]
        label = result["matches"][0]["label"] if result["matches"] else "Unknown"
        ripeness = result.get("ripeness", "N/A")
        class_id = result.get("class_id", 0)
        annotate = result.get("annotate", False)
        
        x_min, y_min, x_max, y_max = bbox
        
        if annotate:
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
            
            # Prepare label with ripeness
            label_with_ripeness = f"{label} | Ripeness: {ripeness}"
            
            # Draw label
            cv2.putText(
                annotated_frame,
                label_with_ripeness,
                (x_min, max(y_min - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            
            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / W_orig
            y_center = (y_min + y_max) / 2 / H_orig
            width = (x_max - x_min) / W_orig
            height = (y_max - y_min) / H_orig
            
            # Append label in YOLO format
            #label_contents.append(f"{class_id} {x_center} {y_center} {width} {height} {label}")
            label_contents.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    # Save the annotated image
    cv2.imwrite(output_image_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for saving
    logging.info(f"[annotate_image] Annotated image saved at: {output_image_path}")
    
    # Save labels in YOLO format
    label_file_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(output_image_path))[0] + ".txt")
    with open(label_file_path, "w") as label_file:
        for content in label_contents:
            label_file.write(content + "\n")
    logging.info(f"[annotate_image] YOLO labels saved at: {label_file_path}")

##############################################################################
# Process and Save Input Image Patches
##############################################################################

def process_and_save_input_image_patches(input_image_path, boxes, patch_save_dir="cropped", patch_size=(100, 100)):
    """
    Extract and save patches from the input image based on detected bounding boxes.
    
    Parameters:
        input_image_path (str): Path to the input image.
        boxes (list): List of bounding boxes in the format [x_min, y_min, x_max, y_max].
        patch_save_dir (str): Directory to save the extracted patches.
        patch_size (tuple): Size of the patch to extract (width, height).
    """
    os.makedirs(patch_save_dir, exist_ok=True)
    logging.info(f"\n[process_and_save_input_image_patches] Loading input image: {input_image_path}")
    image = np.array(Image.open(input_image_path).convert("RGB"))

    for i, bbox in enumerate(boxes):
        # Calculate center of mass for each bounding box
        x_min, y_min, x_max, y_max = map(int, bbox)
        bbox_patch = image[y_min:y_max, x_min:x_max]
        center = calculate_center_of_mass(bbox_patch)

        # Adjust center to the global coordinates
        center_global = (center[0] + x_min, center[1] + y_min)

        # Extract and save the patch for vector database
        extract_and_save_patch(
            image=image,
            center=center_global,
            save_path=patch_save_dir,
            patch_name=f"object_{i}",
            patch_size=patch_size
        )

##############################################################################
# MAIN PIPELINE
##############################################################################
def main_pipeline(
    input_image_path: str,
    reference_images: list,
    groundingdino_config: str,
    groundingdino_weights: str,
    clip_model_dir: str,
    output_folder: str,
    prompt: str = None,
    openai_key: str = None,
    use_lab: bool = False,
    patch_size: tuple = (100, 100),
    label_to_id_map: dict = {},
    target_objects: list = []
):
    """
    Main pipeline for object detection, patch extraction, classification, and saving annotated image and patches.
    """
    # Setup output directories
    processed_images_dir = os.path.join(output_folder, "processed_images")
    images_dir = os.path.join(output_folder, "images")
    labels_dir = os.path.join(output_folder, "labels")
    cropped_dir = os.path.join(output_folder, "cropped")

    for directory in [processed_images_dir, images_dir, labels_dir, cropped_dir]:
        os.makedirs(directory, exist_ok=True)

    logging.info(f"\n[main_pipeline] Loading GroundingDINO model...")
    model = load_groundingdino_model(groundingdino_config, groundingdino_weights)
    
    logging.info(f"\n[main_pipeline] Loading local CLIP model...")
    # Load the CLIP model
    clip_model, processor, device = load_clip_model(clip_model_dir, device=None)
    
    logging.info(f"\n[main_pipeline] Building vector store from reference images...")
    vectorstore = build_reference_vectorstore(reference_images, clip_model, processor, device, use_lab=use_lab, patch_size=patch_size)
    
    # Initialize ChatOpenAI if OpenAI key is provided
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key  # Set the OpenAI API key
        logging.info(f"\n[main_pipeline] Initializing ChatOpenAI model...")
        try:
            chat = ChatOpenAI(model_name="gpt-4", max_tokens=1000)  # Use the specified GPT model
        except Exception as e:
            logging.error(f"[main_pipeline] Failed to initialize ChatOpenAI: {e}")
            chat = None
    else:
        chat = None
        logging.info(f"\n[main_pipeline] OpenAI API key not provided. Skipping GPT-based ripeness assessment.")
    
    logging.info(f"\n[main_pipeline] Segmenting objects in input image...")
    # Load input image
    image_source, image_transformed = load_image(input_image_path)
    
    TEXT_PROMPT = prompt  # Use the passed parameter
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    
    # Detect bounding boxes
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    logging.info(f"[main_pipeline] Detected {len(boxes)} bounding boxes.")
    for i, box in enumerate(boxes):
        logging.info(f"Box {i}: {box}")
    
    # Rescale bounding boxes to original image dimensions
    H_trans, W_trans = image_transformed.shape[1:3]
    H_orig, W_orig, _ = image_source.shape
    scale_x, scale_y = W_orig / W_trans, H_orig / H_trans
    
    rescaled_boxes = []
    for bbox_torch in boxes:
        x_center_norm, y_center_norm, width_norm, height_norm = bbox_torch.tolist()
        x_min_abs = int((x_center_norm - width_norm / 2) * W_orig)
        y_min_abs = int((y_center_norm - height_norm / 2) * H_orig)
        x_max_abs = int((x_center_norm + width_norm / 2) * W_orig)
        y_max_abs = int((y_center_norm + height_norm / 2) * H_orig)
        
        # Validate bounding box
        validated_bbox = validate_bbox([x_min_abs, y_min_abs, x_max_abs, y_max_abs], W_orig, H_orig)
        rescaled_boxes.append(validated_bbox)
    
    logging.info(f"\n[main_pipeline] Saving patches of detected objects...")
    process_and_save_input_image_patches(
        input_image_path=input_image_path,
        boxes=rescaled_boxes,
        patch_save_dir=cropped_dir,
        patch_size=patch_size
    )
    
    # Copy original image to images_dir
    original_image_name = os.path.basename(input_image_path)
    shutil.copy(input_image_path, os.path.join(images_dir, original_image_name))
    logging.info(f"[main_pipeline] Original image copied to: {os.path.join(images_dir, original_image_name)}")
    
    logging.info(f"\n[main_pipeline] Classifying each segmented region...")
    results = classify_segmented_objects(
        image_rgb=image_source,
        boxes=rescaled_boxes,
        clip_model=clip_model,
        processor=processor,
        device=device,
        vectorstore=vectorstore,
        chat=chat,
        top_k=1,  # Changed to top_k=1 as per classification requirement
        use_lab=use_lab,
        cropped_save_dir=cropped_dir,
        patch_size=patch_size,
        label_to_id_map=label_to_id_map,
        target_objects=target_objects
    )
    
    logging.info(f"\n[main_pipeline] Annotating image with bounding boxes and labels...")
    annotate_image(
        image_rgb=image_source,
        results=results,
        output_image_path=os.path.join(processed_images_dir, original_image_name),
        labels_dir=labels_dir,
        W_orig=W_orig,
        H_orig=H_orig
    )

##############################################################################
# Helper Functions
##############################################################################
def validate_bbox(bbox: list, image_width: int, image_height: int) -> list:
    """Ensure bounding box coordinates are within image boundaries."""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(x_min, image_width - 1))
    y_min = max(0, min(y_min, image_height - 1))
    x_max = max(0, min(x_max, image_width - 1))
    y_max = max(0, min(y_max, image_height - 1))
    return [x_min, y_min, x_max, y_max]

def get_class_id(label: str, label_to_id_map: dict) -> int:
    """Map label to a class ID using the provided mapping."""
    return label_to_id_map.get(label, len(label_to_id_map))  # Default to a new ID if label not found

def determine_label_from_config(config_path: str) -> list:
    """
    Read the configuration CSV file and return a list of (image_path, label) tuples.
    
    Parameters:
        config_path (str): Path to the configuration CSV file.
    
    Returns:
        list: List of tuples containing image paths and their corresponding labels.
    """
    reference_images = []
    with open(config_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_filename = row['image_filename'].strip()
            label = row['label'].strip()
            reference_images.append((image_filename, label))
    return reference_images

##############################################################################
# Argument Parser
##############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Ripeness Assessment Script")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration CSV file (conf.txt)."
    )
    parser.add_argument(
        "--reference_images_folder",
        type=str,
        required=True,
        help="Path to the folder containing reference images."
    )
    parser.add_argument(
        "--input_images_folder",
        type=str,
        required=True,
        help="Path to the folder containing input images to be processed."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the output folder where results will be saved."
    )
    parser.add_argument(
        "--groundingdino_config",
        type=str,
        required=True,
        help="Path to the GroundingDINO configuration file."
    )
    parser.add_argument(
        "--groundingdino_weights",
        type=str,
        required=True,
        help="Path to the GroundingDINO weights file."
    )
    parser.add_argument(
        "--clip_model_dir",
        type=str,
        required=True,
        help="Path to the local CLIP model directory."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to the text file containing the custom prompt for GPT-4."
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, GPT-based ripeness assessment will be skipped."
    )
    parser.add_argument(
        "--use_lab",
        action='store_true',
        help="Use LAB color space for embeddings."
    )
    parser.add_argument(
        "--patch_width",
        type=int,
        default=100,
        help="Width of the patch to extract from the center of mass."
    )
    parser.add_argument(
        "--patch_height",
        type=int,
        default=100,
        help="Height of the patch to extract from the center of mass."
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-4",
        help="GPT model to use for ripeness assessment (e.g., 'gpt-4')."
    )
    parser.add_argument(
        "--target_objects",
        nargs='+',
        required=True,
        help="List of target objects to assess (e.g., 'olive fruit' 'apple')."
    )
    return parser.parse_args()

##############################################################################
# MAIN EXECUTION
##############################################################################
def main():
    args = parse_arguments()

    # Read reference images and their labels from config file
    reference_images = determine_label_from_config(args.config_file)
    
    if not reference_images:
        logging.error("[ERROR] No reference images found in the configuration file.")
        exit(1)

    # Prepend the reference_images_folder path to each image filename
    reference_images = [(os.path.join(args.reference_images_folder, img), label) for img, label in reference_images]

    # Create a mapping from labels to unique class IDs
    unique_labels = sorted(list(set([label for _, label in reference_images])))
    label_to_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    logging.info(f"[Main] Label to Class ID Mapping: {label_to_id_map}")

    # Read custom prompt if provided; else, use target_objects as default prompt
    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, "r", encoding='utf-8') as prompt_f:
                custom_prompt = prompt_f.read()
            logging.info(f"[Main] Custom prompt loaded from {args.prompt_file}")
        else:
            logging.error(f"[ERROR] Prompt file not found: {args.prompt_file}")
            custom_prompt = " ".join(args.target_objects)  # Fallback to target_objects as prompt
    else:
        custom_prompt = " ".join(args.target_objects)  # Use the target_objects as default prompt

    # Define patch size
    patch_size = (args.patch_width, args.patch_height)
    logging.info(f"[Main] Using patch size: Width={patch_size[0]}, Height={patch_size[1]}")

    # Iterate over each input image and process
    for root, dirs, files in os.walk(args.input_images_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_image_path = os.path.join(root, file)
                logging.info(f"\n[Main] Processing image: {input_image_path}")
                main_pipeline(
                    input_image_path=input_image_path,
                    reference_images=reference_images,
                    groundingdino_config=args.groundingdino_config,
                    groundingdino_weights=args.groundingdino_weights,
                    clip_model_dir=args.clip_model_dir,
                    output_folder=args.output_folder,
                    prompt=custom_prompt,  # Pass the resolved prompt
                    openai_key=args.openai_key,
                    use_lab=args.use_lab,
                    patch_size=patch_size,
                    label_to_id_map=label_to_id_map,
                    target_objects=args.target_objects
                )

##############################################################################
# Entry Point
##############################################################################
if __name__ == "__main__":
    main()
