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
def load_clip_model(device: torch.device = None) -> tuple:
    """Load the local CLIP model."""
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logging.info(f"[load_clip_model] Using device: {device}")

    LOCAL_CLIP_DIR = "/Users/macbookpro/Documents/Projects/kws/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(LOCAL_CLIP_DIR)
    clip_model.to(device)  # Move model to device
    processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_DIR)
    return clip_model, processor, device

##############################################################################
# Vector Store Functions
##############################################################################
def build_reference_vectorstore(reference_images: list, clip_model: CLIPModel, processor: CLIPProcessor, device: torch.device, use_lab: bool = False, patch_save_dir: str = "patches/vector_db") -> FAISS:
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
        patch = extract_center_patch(np_image, center, patch_size=(50, 50))
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
# OpenAI GPT-4 Integration via LangChain
##############################################################################
def get_ripeness_from_gpt(chat: ChatOpenAI, image_path: str, vector_db_label: str) -> str:
    """
    Send the entire detected region image and vector database label to GPT-4 to get the ripeness score.

    Parameters:
        chat (ChatOpenAI): Initialized LangChain ChatOpenAI instance.
        image_path (str): Path to the saved image of the detected object (entire region).
        vector_db_label (str): Label obtained from the vector database.

    Returns:
        str: Ripeness score based on the Jaén Scale.
    """
    # Encode the image to base64
    mime_type = get_mime_type(image_path)
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

    # Prepare the prompt
    prompt = (
        "You are an olive fruit specialist. Based on the skin color (Jaén Scale) and the information from the vector database, "
        "what is the ripeness of this olive fruit?\n\n"
        "Scale from 0 to 7, where:\n"
        "0: Intense green.\n"
        "1: Yellowish green.\n"
        "2: Purple spots.\n"
        "3: Purple on more than 50% of the fruit.\n"
        "4: Purple/black on the surface, pulp still green.\n"
        "5: Black with partially purple pulp.\n"
        "6: Black with fully purple pulp.\n"
        "7: Black with dark pulp all the way to the stone.\n\n"
        f"Vector Database Label: {vector_db_label}\n\n"
        "Please analyze the attached image and provide the ripeness score based on the Jaén Scale. "
        "Return Class 0 - Green, Class 1 - Green Yellow, Class 2 - <50% red, Class 3 - > 50% red, Class 4 - red, Class 5 - purple/black, directly."
    )

    # Prepare the message with the encoded image and the custom prompt
    message_content = [
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}  # Dynamic MIME type
        },
    ]

    # Create a HumanMessage with the prepared content
    human_message = HumanMessage(content=message_content)

    # Invoke the model
    try:
        output = chat.invoke([human_message])
        ripeness = output.content.strip()

        logging.info(f"Ripeness Score: {ripeness}")
        print ("[get_ripeness_from_gpt] GPT Output:", ripeness)
    except Exception as e:
        logging.error(f"[ERROR] OpenAI API call failed: {e}")
        ripeness = "Ripeness assessment unavailable."

    return ripeness

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
    chat: ChatOpenAI,
    top_k: int = 1,
    use_lab: bool = False
) -> list:
    """Classify each segmented object by matching patches against the reference vectorstore and assessing ripeness via GPT."""
    results = []
    for j, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox
        center_global = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        
        # Extract patch for vector database comparison
        patch = extract_center_patch(image_rgb, center_global, patch_size=(100, 100))
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
        
        # Save the detected patch image
        detected_patches_dir = "patches/detected_patches"
        os.makedirs(detected_patches_dir, exist_ok=True)
        patch_save_path = os.path.join(detected_patches_dir, f"detected_patch_{j}.png")
        patch_pil.save(patch_save_path)
        logging.info(f"[INFO] Detected patch image saved at: {patch_save_path}")
        
        # Extract the entire detected region for GPT ripeness assessment
        detected_objects_dir = "patches/detected_objects_full"
        os.makedirs(detected_objects_dir, exist_ok=True)
        detected_region = image_rgb[y_min:y_max, x_min:x_max]
        detected_region_pil = Image.fromarray(detected_region)
        detected_region_save_path = os.path.join(detected_objects_dir, f"detected_object_full_{j}.png")
        detected_region_pil.save(detected_region_save_path)
        logging.info(f"[INFO] Detected full region image saved at: {detected_region_save_path}")
        
        # Get ripeness from GPT-4 using the entire detected region
        vector_db_label = matches[0]["label"] if matches else "Unknown"
        print ("[classify_segmented_objects] Vector DB Label:", vector_db_label)
        ripeness = get_ripeness_from_gpt(chat, detected_region_save_path, vector_db_label)
        print ("[classify_segmented_objects] GPT Ripeness Score:", ripeness)
        logging.info(f"Box #{j} - Ripeness Score: {ripeness}")
        
        results.append({
            "bbox": bbox,
            "matches": matches,
            "ripeness": ripeness
        })
    return results

##############################################################################
# Export Annotations to YOLO Format
##############################################################################
def export_to_yolo(output_txt_path: str, boxes: list, class_ids: list, image_width: int, image_height: int):
    """
    Export bounding boxes and class IDs to YOLO format.

    Parameters:
        output_txt_path (str): Path to the output YOLO annotation file.
        boxes (list): List of bounding boxes in the format [x_min, y_min, x_max, y_max].
        class_ids (list): List of class IDs corresponding to each bounding box.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    with open(output_txt_path, 'w') as f:
        for bbox, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = bbox
            # Calculate YOLO format values
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            # Ensure values are between 0 and 1
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    logging.info(f"[INFO] YOLO annotations saved at: {output_txt_path}")

##############################################################################
# Process and Save Input Image Patches
##############################################################################

def process_and_save_input_image_patches(input_image_path, boxes, patch_save_dir="patches/detected_objects", patch_size=(100, 100)):
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

        # Extract and save the patch
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
def main_pipeline(input_image_path: str, reference_images: list, groundingdino_config: str, groundingdino_weights: str, use_lab: bool = False):
    """
    Main pipeline for object detection, patch extraction, classification, ripeness assessment, and exporting annotations to YOLO format.
    """
    logging.info(f"\n[main_pipeline] Loading GroundingDINO model...")
    model = load_groundingdino_model(groundingdino_config, groundingdino_weights)
    
    logging.info(f"\n[main_pipeline] Loading local CLIP model...")
    clip_model, processor, device = load_clip_model()
    
    logging.info(f"\n[main_pipeline] Building vector store from reference images...")
    vectorstore = build_reference_vectorstore(reference_images, clip_model, processor, device, use_lab=use_lab)
    
    logging.info(f"\n[main_pipeline] Initializing ChatOpenAI model...")
    chat = ChatOpenAI(model="gpt-4o", max_tokens=10000)  # Corrected model name
    
    logging.info(f"\n[main_pipeline] Segmenting objects in input image...")
    # Load input image
    image_source, image_transformed = load_image(input_image_path)
    
    TEXT_PROMPT = "olive fruit"
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
        patch_save_dir="patches/detected_objects",
        patch_size=(100, 100)
    )
    
    logging.info(f"\n[main_pipeline] Classifying each segmented region and assessing ripeness...")
    results = classify_segmented_objects(
        image_rgb=image_source,
        boxes=rescaled_boxes,
        clip_model=clip_model,
        processor=processor,
        device=device,
        vectorstore=vectorstore,
        chat=chat,
        top_k=1,  # Changed to top_k=1 as per classification requirement
        use_lab=use_lab
    )
    
    # Mapping from label to class ID
    label_to_class_id = {
        "Class 0 - Green": 0,
        "Class 1 - Green Yellow": 1,
        "Class 2 - <50% red": 2,
        "Class 3 - > 50% red": 3,
        "Class 4 - red": 4,
        "Class 5 - purple/black": 5
    }
    
    # Prepare data for YOLO export
    class_ids = []
    for result in results:
        label = result["matches"][0]["label"] if result["matches"] else "Unknown"
        class_id = label_to_class_id.get(label, -1)  # Assign -1 for unknown classes
        if class_id == -1:
            logging.warning(f"[WARNING] Unknown label '{label}' encountered. Skipping in YOLO export.")
            continue  # Skip unknown classes
        class_ids.append(class_id)
    
    # Get image dimensions
    image_height, image_width, _ = image_source.shape
    
    # Define YOLO annotation file path
    yolo_annotation_path = os.path.splitext(os.path.basename(input_image_path))[0] + ".txt"
    
    # Export to YOLO format
    export_to_yolo(
        output_txt_path=yolo_annotation_path,
        boxes=rescaled_boxes,
        class_ids=class_ids,
        image_width=image_width,
        image_height=image_height
    )
    
    logging.info(f"\n[main_pipeline] Annotating image with bounding boxes, labels, and ripeness scores...")
    annotated_frame = image_source.copy()
    for result, rescaled_bbox in zip(results, rescaled_boxes):
        x_min, y_min, x_max, y_max = rescaled_bbox
        label = result["matches"][0]["label"] if result["matches"] else "Unknown"
        ripeness = result.get("ripeness", "N/A")

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Prepare label with ripeness
        label_with_ripeness = f"{label} | Ripeness: {ripeness}"

        # Draw label
        cv2.putText(
            annotated_frame,
            label_with_ripeness,
            (x_min, max(y_min - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Save the annotated image
    output_path = "output_annotated_with_classes.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for saving
    logging.info(f"[main_pipeline] Annotated image with classes and ripeness saved at: {output_path}")

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

##############################################################################
# Export Annotations to YOLO Format
##############################################################################
def export_to_yolo(output_txt_path: str, boxes: list, class_ids: list, image_width: int, image_height: int):
    """
    Export bounding boxes and class IDs to YOLO format.

    Parameters:
        output_txt_path (str): Path to the output YOLO annotation file.
        boxes (list): List of bounding boxes in the format [x_min, y_min, x_max, y_max].
        class_ids (list): List of class IDs corresponding to each bounding box.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    with open(output_txt_path, 'w') as f:
        for bbox, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = bbox
            # Calculate YOLO format values
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            # Ensure values are between 0 and 1
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    logging.info(f"[INFO] YOLO annotations saved at: {output_txt_path}")

##############################################################################
# Main Execution
##############################################################################
if __name__ == "__main__":
    # Define reference images with their labels
    reference_images = [
        ("class_0_img1.png", "Class 0 - Green"),
        ("class_0_img2.png", "Class 0 - Green"),
        ("class_0_img3.png", "Class 0 - Green"),
        ("class_0_img4.png", "Class 0 - Green"),
        ("class_1_img1.png", "Class 1 - Green Yellow"),
        ("class_1_img2.png", "Class 1 - Green Yellow"),
        ("class_1_img3.png", "Class 1 - Green Yellow"),
        ("class_1_img4.png", "Class 1 - Green Yellow"),
        ("class_2_img1.png", "Class 2 - <50% red"),
        ("class_2_img2.png", "Class 2 - <50% red"),
        ("class_2_img3.png", "Class 2 - <50% red"),
        ("class_2_img4.png", "Class 2 - <50% red"),
        ("class_3_img1.png", "Class 3 - > 50% red"),
        ("class_3_img2.png", "Class 3 - > 50% red"),
        ("class_3_img3.png", "Class 3 - > 50% red"),
        ("class_3_img4.png", "Class 3 - > 50% red"),
        ("class_4_img1.png", "Class 4 - red"),
        ("class_4_img2.png", "Class 4 - red"),
        ("class_4_img3.png", "Class 4 - red"),
        ("class_4_img4.png", "Class 4 - red"),
        ("class_5_img1.png", "Class 5 - purple/black"),
        ("class_5_img2.png", "Class 5 - purple/black"),
        ("class_5_img3.png", "Class 5 - purple/black"),
        ("class_5_img4.png", "Class 5 - purple/black"),
    ]
    
    # Define GroundingDINO configuration and weights paths
    groundingdino_config = "/Users/macbookpro/Documents/Projects/kws/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_weights = "/Users/macbookpro/Documents/Projects/kws/weights/groundingdino_swint_ogc.pth"

    # Define input image path
    input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3111_jpeg.rf.b4156c5076b9dd9827809c01ffcca46a.jpg"

    # Run the pipeline
    main_pipeline(
        input_image_path=input_image_path,
        reference_images=reference_images,
        groundingdino_config=groundingdino_config,
        groundingdino_weights=groundingdino_weights,
        use_lab=True
    )
