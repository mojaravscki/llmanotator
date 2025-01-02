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
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.docstore.document import Document
from scipy.ndimage import center_of_mass
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
import io
import argparse
import shutil
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional
from dataclasses_json import dataclass_json
import supervision as sv
from supervision import Detections
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from segment_anything import sam_model_registry, SamPredictor
import json  # For JSON output
import pickle  # For pickle serialization
import re  # ### GPT CLASS PARSER

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
# COCO Data Classes
##############################################################################
@dataclass_json
@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str

@dataclass_json
@dataclass
class COCOImage:
    id: int
    width: int
    height: int
    file_name: str
    license: Optional[int] = None
    date_captured: Optional[str] = None
    coco_url: Optional[str] = None
    flickr_url: Optional[str] = None

@dataclass_json
@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: List[float]
    iscrowd: int = 0

@dataclass_json
@dataclass
class COCOLicense:
    id: int
    name: str
    url: str

@dataclass_json
@dataclass
class COCOJson:
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]
    licenses: Optional[List[COCOLicense]] = None

##############################################################################
# Utility Functions
##############################################################################
def get_mime_type(image_path: str) -> str:
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
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

def calculate_center_of_mass(image: np.ndarray) -> tuple:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized = gray / 255.0
    center = center_of_mass(normalized)
    return int(center[1]), int(center[0])  # (x, y)

def extract_center_patch(image: np.ndarray, center: tuple, patch_size: tuple) -> np.ndarray:
    cx, cy = center
    half_w, half_h = patch_size[0] // 2, patch_size[1] // 2

    x_start, x_end = max(cx - half_w, 0), min(cx + half_w, image.shape[1])
    y_start, y_end = max(cy - half_h, 0), min(cy + half_h, image.shape[0])
    patch = image[y_start:y_end, x_start:x_end]

    if patch.size == 0:
        logging.error(f"[ERROR] Empty patch extracted at center: {center}")
        raise ValueError("Empty patch extracted. Check center coordinates.")
    return patch

def extract_and_save_patch(image: np.ndarray, center: tuple, save_path: str, patch_name: str, patch_size: tuple):
    patch = extract_center_patch(image, center, patch_size)
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{patch_name}.png")
    cv2.imwrite(save_file, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    logging.info(f"[INFO] Patch saved at: {save_file}")
    return patch

##############################################################################
# CLIP Functions
##############################################################################
def resize_image(image: Image.Image, max_size=(800, 800)) -> Image.Image:
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def encode_image(image_path: str) -> str:
    mime_type = get_mime_type(image_path)
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext in [".jpg", ".jpeg"]:
        image_format = "JPEG"
    elif file_ext == ".png":
        image_format = "PNG"
    else:
        image_format = "PNG"

    with Image.open(image_path).convert("RGB") as img:
        img = resize_image(img)
        buffered = io.BytesIO()
        img.save(buffered, format=image_format)

    encoded_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_str}"

def get_image_embedding(image: Image.Image,
                        clip_model: CLIPModel,
                        processor: CLIPProcessor,
                        device: torch.device,
                        use_lab: bool = False) -> np.ndarray:
    if use_lab:
        np_image = np.array(image)
        np_image_lab = convert_to_lab(np_image)
        image = Image.fromarray(np_image_lab)

    if device.type in ['cuda', 'mps']:
        autocast_type = device.type
        with torch.amp.autocast(autocast_type):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
    else:
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
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logging.info(f"[load_groundingdino_model] Using device: {device}")
    model = load_model(config_path, weights_path)
    model.to(device)
    return model

##############################################################################
# CLIP Loader
##############################################################################
def load_clip_model(clip_model_dir: str, device: torch.device = None) -> tuple:
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logging.info(f"[load_clip_model] Using device: {device}")

    clip_model = CLIPModel.from_pretrained(clip_model_dir)
    clip_model.to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_dir)
    return clip_model, processor, device

##############################################################################
# Vector Store Functions
##############################################################################
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

def build_reference_vectorstore(
    reference_images: list,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    use_lab: bool = False,
    patch_save_dir: str = "output/vector_db",
    patch_size: tuple = (100, 100),
    persistent: bool = False,
    vector_db_dir: str = "output/vector_db"
) -> FAISS:
    if persistent:
        index_file = os.path.join(vector_db_dir, "faiss_index.faiss")
        embeddings_file = os.path.join(vector_db_dir, "embeddings.pkl")
        if os.path.exists(index_file) and os.path.exists(embeddings_file):
            logging.info("[build_reference_vectorstore] Loading existing FAISS index and embeddings from disk.")
            try:
                class NumpyArrayEmbeddings(Embeddings):
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        return []
                    def embed_query(self, text: str) -> List[float]:
                        return []
                embedding_function = NumpyArrayEmbeddings()
                vectorstore = FAISS.load_local(vector_db_dir, embedding=embedding_function)
                logging.info(f"[build_reference_vectorstore] FAISS index loaded from {vector_db_dir}.")
                return vectorstore
            except Exception as e:
                logging.error(f"[build_reference_vectorstore] Failed to load FAISS index: {e}")
                logging.info("[build_reference_vectorstore] Rebuilding vector store...")

    docs = []
    embeddings_list = []

    def augment_image(image: Image.Image):
        augmented_images = [image]
        augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        augmented_images.append(image.rotate(-90, expand=True))
        augmented_images.append(image.rotate(90, expand=True))
        enhancer = ImageEnhance.Color(image)
        for sf in [0.75, 1.25]:
            augmented_images.append(enhancer.enhance(sf))
        return augmented_images

    os.makedirs(patch_save_dir, exist_ok=True)

    for i, (img_path, label) in enumerate(reference_images):
        if not os.path.exists(img_path):
            logging.warning(f"[WARNING] File not found: {img_path} — skipping.")
            continue

        logging.info(f"\n[build_reference_vectorstore] Loading reference #{i}: {img_path}")
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"[ERROR] Failed to open image {img_path}: {e}")
            continue
        np_image = np.array(pil_img)

        try:
            center = calculate_center_of_mass(np_image)
            patch = extract_center_patch(np_image, center, patch_size=patch_size)
            patch_pil = Image.fromarray(patch)
        except ValueError as ve:
            logging.warning(f"[WARNING] {ve} — skipping {img_path}.")
            continue

        original_patch_save_path = os.path.join(
            patch_save_dir,
            f"{os.path.splitext(os.path.basename(img_path))[0]}_crop{i}_original.png"
        )
        try:
            patch_pil.save(original_patch_save_path)
            logging.info(f"[INFO] Original patch saved at: {original_patch_save_path}")
        except Exception as e:
            logging.error(f"[ERROR] Failed to save {original_patch_save_path}: {e}")
            continue

        try:
            emb = get_image_embedding(patch_pil, clip_model, processor, device, use_lab=use_lab)
            emb = emb.astype(np.float32).tolist()
        except Exception as e:
            logging.error(f"[ERROR] Embedding error for {original_patch_save_path}: {e}")
            continue

        doc = Document(page_content=label, metadata={"label": label})
        docs.append(doc)
        embeddings_list.append(emb)

        aug_imgs = augment_image(patch_pil)
        for aug_idx, aug_img in enumerate(aug_imgs):
            aug_patch_filename = (
                f"{os.path.splitext(os.path.basename(img_path))[0]}_crop{i}_aug{aug_idx}"
            )
            aug_patch_save_path = os.path.join(patch_save_dir, f"{aug_patch_filename}.png")
            try:
                aug_img.save(aug_patch_save_path)
            except Exception as e:
                logging.error(f"[ERROR] Failed to save augmented patch {aug_patch_save_path}: {e}")
                continue

            try:
                emb_aug = get_image_embedding(aug_img, clip_model, processor, device, use_lab=use_lab)
                emb_aug = emb_aug.astype(np.float32).tolist()
            except Exception as e:
                logging.error(f"[ERROR] Embedding error for {aug_patch_save_path}: {e}")
                continue

            doc_aug = Document(page_content=f"{label} (Augmentation {aug_idx})", metadata={"label": label})
            docs.append(doc_aug)
            embeddings_list.append(emb_aug)

    if not docs:
        raise ValueError("[ERROR] No valid reference images found!")

    class NumpyArrayEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return []
        def embed_query(self, text: str) -> List[float]:
            return []

    embedding_function = NumpyArrayEmbeddings()
    text_embedding_tuples = [(doc.page_content, emb) for doc, emb in zip(docs, embeddings_list)]
    all_metadatas = [doc.metadata for doc in docs]

    logging.info(f"\n[build_reference_vectorstore] Building FAISS from {len(text_embedding_tuples)} items.")
    try:
        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_tuples,
            embedding=embedding_function,
            metadatas=all_metadatas,
        )
    except Exception as e:
        logging.error(f"[ERROR] Failed to build vector store: {e}")
        raise

    if persistent:
        try:
            vectorstore.save_local(vector_db_dir)
            logging.info(f"[build_reference_vectorstore] Saved FAISS index in {vector_db_dir}.")
        except Exception as e:
            logging.error(f"[ERROR] Failed to save FAISS index or embeddings: {e}")

    return vectorstore

##############################################################################
# GPT Helper Functions
##############################################################################
def first_gpt_call(chat: ChatOpenAI, image_path: str, prompt_text: str) -> str:
    base64_image = encode_image(image_path)
    message_content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": base64_image}},
    ]
    human_message = HumanMessage(content=message_content)
    try:
        response = chat.invoke([human_message]).content.strip()
        logging.info(f"######### GPT FIRST CALL RESPONSE: {response}")
        return response
    except Exception as e:
        logging.error(f"[first_gpt_call] ChatGPT call failed: {e}")
        return "Preliminary classification unavailable"

def second_gpt_call(chat: ChatOpenAI,
                    image_path: str,
                    vector_db_label: str,
                    first_gpt_result: str,
                    final_prompt: str) -> str:
    base64_image = encode_image(image_path)
    combined_prompt = (
        f"{final_prompt}\n\n"
        f"Vector Database Label: {vector_db_label}\n\n"
        f"First GPT Result: {first_gpt_result}\n\n"
        f"Please merge the above info and finalize the ripeness classification."
        f"Return Directly if it Class 0 - Green, Class 1 - Green Yellow, Class 2 - <50% red, Class 3 - > 50% red, Class 4 - red, Class 5 - purple/black."
    )

    message_content = [
        {"type": "text", "text": combined_prompt},
        {"type": "image_url", "image_url": {"url": base64_image}},
    ]
    human_message = HumanMessage(content=message_content)
    try:
        response = chat.invoke([human_message]).content.strip()
        logging.info(f"######### GPT SECOND CALL RESPONSE: {response}")
        return response
    except Exception as e:
        logging.error(f"[second_gpt_call] ChatGPT call failed: {e}")
        return "Final classification unavailable"

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
    patch_save_dir: str = "patches",
    patch_size: tuple = (100, 100),
    label_to_id_map: dict = {},
    target_objects: list = [],
    ripeness_prompt: str = ""
) -> list:
    results = []
    original_filename = "unknown"

    for j, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox
        center_global = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        
        # 1) Extract center patch
        try:
            patch = extract_center_patch(image_rgb, center_global, patch_size=patch_size)
            patch_pil = Image.fromarray(patch)
        except ValueError as ve:
            logging.warning(f"Skipping box #{j}: {ve}")
            continue
        
        # 2) Vector DB embedding + search
        try:
            emb = get_image_embedding(patch_pil, clip_model, processor, device, use_lab=use_lab)
            emb = emb.astype(np.float32).reshape(-1)
            docs = vectorstore.similarity_search_by_vector(emb, k=top_k)
        except Exception as e:
            logging.error(f"Box #{j} error in Vector DB search: {e}")
            docs = []

        matches = []
        for k_idx, doc in enumerate(docs):
            label = doc.metadata.get("label", "Unknown")
            matches.append({"label": label, "distance": "N/A"})
            logging.info(f"################# Vectordatabase: Box #{j} - Match {k_idx+1}: {label}")

        vector_db_label = matches[0]["label"] if matches else "Unknown"

        # 3) Save patch to file for GPT usage
        patch_filename = f"{os.path.splitext(original_filename)[0]}_patch{j}.png"
        patch_path = os.path.join(patch_save_dir, patch_filename)
        try:
            patch_pil.save(patch_path)
        except Exception as e:
            logging.error(f"Cannot save patch for box #{j}: {e}")
            patch_path = None

        # 4) If we have GPT, do the two-step approach
        preliminary_result = "N/A"
        final_classification = "N/A"
        orchestrator_decision = {"annotate": False}

        if chat and patch_path and ripeness_prompt:
            # Step A
            preliminary_result = first_gpt_call(chat, patch_path, ripeness_prompt)
            # Step B
            final_classification = second_gpt_call(chat, patch_path, vector_db_label,
                                                  preliminary_result, ripeness_prompt)
            ripeness = final_classification
            orchestrator_decision["annotate"] = True
        else:
            # fallback
            ripeness = vector_db_label

        class_id = label_to_id_map.get(vector_db_label, len(label_to_id_map))
        results.append({
            "bbox": bbox,
            "matches": matches,
            "ripeness": ripeness,
            "class_id": class_id,  # Vector DB ID
            "annotate": orchestrator_decision.get("annotate", False)
        })

    return results

##############################################################################
# ### GPT CLASS PARSER
##############################################################################
def parse_gpt_class(gpt_string: str) -> int:
    """
    Attempt to extract "Class N" from the GPT final string.
    If not found, default to 0.
    """
    match = re.search(r'[Cc]lass\s+(\d+)', gpt_string)
    if match:
        return int(match.group(1))
    return 0

##############################################################################
# SAM Model Loader
##############################################################################
from typing import Tuple

def load_sam_model(model_type: str = "vit_h", checkpoint_path: str = "sam_vit_h.pth") -> Tuple[SamPredictor, torch.device]:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("[load_sam_model] Using MPS (Apple Silicon) backend.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("[load_sam_model] Using CUDA backend.")
    else:
        device = torch.device("cpu")
        logging.info("[load_sam_model] Using CPU backend.")
    
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        logging.info(f"[load_sam_model] SAM model '{model_type}' loaded successfully.")
    except Exception as e:
        logging.error(f"[load_sam_model] Failed to load SAM model: {e}")
        raise
    
    try:
        mask_predictor = SamPredictor(sam)
        logging.info("[load_sam_model] SAM Predictor initialized successfully.")
    except Exception as e:
        logging.error(f"[load_sam_model] Failed to init SAM Predictor: {e}")
        raise
    
    return mask_predictor, device

##############################################################################
# Annotation Module
##############################################################################
def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.squeeze().tolist()
        if isinstance(contour[0], list):
            polygon = [coord for point in contour for coord in point]
            polygons.append(polygon)
    return polygons

def annotate_image(
    image_rgb: np.ndarray,
    results: list,
    output_image_path: str,
    labels_vdb_dir: str,
    labels_orchestrator_dir: str,
    labels_vdb_polygon_dir: str,
    labels_orchestrator_polygon_dir: str,
    W_orig: int,
    H_orig: int,
    mask_predictor: SamPredictor,
    device: torch.device,
    label_to_id_map: dict
):
    """
    ### MODIFIED FOR GPT:
      - vdb_yolo/polygon => uses vector DB label, "class_id_vdb".
      - orchestrator_yolo/polygon => uses GPT final classification => parse it into a numeric ID.
      - bounding box text in processed_images => GPT final classification.
    """
    annotated_frame = image_rgb.copy()
    label_vdb_contents = []
    label_orchestrator_contents = []
    
    coco_annotations_vdb = []
    coco_annotations_orchestrator = []
    annotation_id_vdb = 1
    annotation_id_orchestrator = 1
    image_id = 1
    
    categories = [{"id": v, "name": k, "supercategory": "object"} for k, v in label_to_id_map.items()]
    
    for result in results:
        bbox = result["bbox"]
        vector_db_label = (result["matches"][0]["label"]
                           if result["matches"] else "Unknown")
        # 'ripeness' might be GPT final classification or vector_db_label fallback
        ripeness = result.get("ripeness", "N/A")

        # ### Vector DB Class ID
        class_id_vdb = label_to_id_map.get(vector_db_label, len(label_to_id_map))

        # ### GPT Class ID => parse from final string (or fallback)
        class_id_orchestrator = parse_gpt_class(ripeness)
        if class_id_orchestrator == 0:
            # if we can't parse a GPT class, fallback:
            class_id_orchestrator = class_id_vdb

        annotate = result.get("annotate", False)
        
        x_min, y_min, x_max, y_max = bbox
        
        # =======================
        # 1) Vector DB YOLO
        # =======================
        x_center_vdb = (x_min + x_max) / 2 / W_orig
        y_center_vdb = (y_min + y_max) / 2 / H_orig
        width_vdb = (x_max - x_min) / W_orig
        height_vdb = (y_max - y_min) / H_orig
        label_vdb_contents.append(
            f"{class_id_vdb} {x_center_vdb} {y_center_vdb} {width_vdb} {height_vdb}"
        )
        
        # =======================
        # Vector DB polygon
        # =======================
        box_vdb = np.array([x_min, y_min, x_max, y_max])
        mask_predictor.set_image(image_rgb)
        try:
            masks_vdb, scores_vdb, logits_vdb = mask_predictor.predict(
                box=box_vdb, multimask_output=False
            )
            mask_vdb = masks_vdb[0]
            polygons_vdb = mask_to_polygon(mask_vdb)
        except Exception as e:
            logging.error(
                f"[annotate_image] SAM prediction failed (VDB) at {bbox}: {e}"
            )
            polygons_vdb = []
        
        if polygons_vdb:
            area_vdb = float(np.sum(mask_vdb))
            coco_annotation_vdb = COCOAnnotation(
                id=annotation_id_vdb,
                image_id=image_id,
                category_id=class_id_vdb,  # use the vector DB ID
                segmentation=polygons_vdb,
                area=area_vdb,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                iscrowd=0
            )
            coco_annotations_vdb.append(coco_annotation_vdb)
            annotation_id_vdb += 1
        else:
            logging.warning(f"No polygon for Vector DB at bbox {bbox}.")

        # =======================
        # 2) Orchestrator YOLO
        # =======================
        # If GPT says we should annotate, do so:
        if annotate:
            # We'll store the GPT final classification as text
            x_center_orchestrator = x_center_vdb
            y_center_orchestrator = y_center_vdb
            width_orchestrator = width_vdb
            height_orchestrator = height_vdb
            # bounding box text is the GPT final classification
            label_orchestrator = ripeness

            label_orchestrator_contents.append(
                f"{class_id_orchestrator} "
                f"{x_center_orchestrator} {y_center_orchestrator} "
                f"{width_orchestrator} {height_orchestrator}"
            )
            
            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )
            # Draw label text
            cv2.putText(
                annotated_frame,
                label_orchestrator,  # GPT final classification
                (x_min, max(y_min - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            
            # Orchestrator polygon
            box_orchestrator = np.array([x_min, y_min, x_max, y_max])
            mask_predictor.set_image(image_rgb)
            try:
                masks_orchestrator, _, _ = mask_predictor.predict(
                    box=box_orchestrator, multimask_output=False
                )
                mask_orchestrator = masks_orchestrator[0]
                polygons_orchestrator = mask_to_polygon(mask_orchestrator)
            except Exception as e:
                logging.error(
                    f"[annotate_image] SAM prediction failed (Orchestrator) at {bbox}: {e}"
                )
                polygons_orchestrator = []
            
            if polygons_orchestrator:
                area_orchestrator = float(np.sum(mask_orchestrator))
                coco_annotation_orchestrator = COCOAnnotation(
                    id=annotation_id_orchestrator,
                    image_id=image_id,
                    category_id=class_id_orchestrator,  # GPT-based ID
                    segmentation=polygons_orchestrator,
                    area=area_orchestrator,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    iscrowd=0
                )
                coco_annotations_orchestrator.append(coco_annotation_orchestrator)
                annotation_id_orchestrator += 1
            else:
                logging.warning(f"No polygon for Orchestrator at bbox {bbox}.")

    # Save the annotated image
    try:
        cv2.imwrite(
            output_image_path,
            cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        )
        logging.info(f"[annotate_image] Annotated image saved: {output_image_path}")
    except Exception as e:
        logging.error(f"[annotate_image] Failed to save {output_image_path}: {e}")

    # =======================
    # YOLO labels (Vector DB)
    # =======================
    try:
        os.makedirs(labels_vdb_dir, exist_ok=True)
        label_vdb_file_path = os.path.join(
            labels_vdb_dir,
            os.path.splitext(os.path.basename(output_image_path))[0] + ".txt"
        )
        with open(label_vdb_file_path, "w") as f:
            for content in label_vdb_contents:
                f.write(content + "\n")
        logging.info(f"[annotate_image] YOLO (VDB) saved: {label_vdb_file_path}")
    except Exception as e:
        logging.error(f"[annotate_image] Failed to save YOLO (VDB): {e}")

    # =======================
    # YOLO labels (Orchestrator)
    # =======================
    try:
        os.makedirs(labels_orchestrator_dir, exist_ok=True)
        label_orchestrator_file_path = os.path.join(
            labels_orchestrator_dir,
            os.path.splitext(os.path.basename(output_image_path))[0] + ".txt"
        )
        with open(label_orchestrator_file_path, "w") as f:
            for content in label_orchestrator_contents:
                f.write(content + "\n")
        logging.info(f"[annotate_image] YOLO (Orchestrator) saved: {label_orchestrator_file_path}")
    except Exception as e:
        logging.error(f"[annotate_image] Failed to save YOLO (Orchestrator): {e}")

    # =======================
    # COCO polygon (Vector DB)
    # =======================
    if coco_annotations_vdb:
        coco_annotation_data_vdb = {
            "images": [{
                "id": image_id,
                "width": W_orig,
                "height": H_orig,
                "file_name": os.path.basename(output_image_path)
            }],
            "annotations": [anno.to_dict() for anno in coco_annotations_vdb],
            "categories": categories
        }
        
        try:
            os.makedirs(labels_vdb_polygon_dir, exist_ok=True)
            coco_output_path_vdb = os.path.join(
                labels_vdb_polygon_dir,
                os.path.splitext(os.path.basename(output_image_path))[0] + "_polygon_vdb.json"
            )
            with open(coco_output_path_vdb, "w") as cf:
                json.dump(coco_annotation_data_vdb, cf, indent=4)
            logging.info(f"[annotate_image] COCO (VDB) saved: {coco_output_path_vdb}")
        except Exception as e:
            logging.error(f"[annotate_image] Failed to save COCO (VDB): {e}")
    else:
        logging.info(f"[annotate_image] No COCO polygons for Vector DB in {output_image_path}.")

    # =======================
    # COCO polygon (Orchestrator)
    # =======================
    if coco_annotations_orchestrator:
        coco_annotation_data_orch = {
            "images": [{
                "id": image_id,
                "width": W_orig,
                "height": H_orig,
                "file_name": os.path.basename(output_image_path)
            }],
            "annotations": [anno.to_dict() for anno in coco_annotations_orchestrator],
            "categories": categories
        }
        
        try:
            os.makedirs(labels_orchestrator_polygon_dir, exist_ok=True)
            coco_output_path_orch = os.path.join(
                labels_orchestrator_polygon_dir,
                os.path.splitext(os.path.basename(output_image_path))[0] + "_polygon_orchestrator.json"
            )
            with open(coco_output_path_orch, "w") as cf:
                json.dump(coco_annotation_data_orch, cf, indent=4)
            logging.info(f"[annotate_image] COCO (Orchestrator) saved: {coco_output_path_orch}")
        except Exception as e:
            logging.error(f"[annotate_image] Failed to save COCO (Orchestrator): {e}")
    else:
        logging.info(f"[annotate_image] No COCO polygons for Orchestrator in {output_image_path}.")

##############################################################################
# Process and Save Input Image Patches
##############################################################################
def process_and_save_input_image_patches(
    input_image_path: str,
    boxes: list,
    original_filename: str,
    cropped_save_dir: str = "cropped",
    patch_save_dir: str = "patches",
    patch_size: tuple = (100, 100)
):
    os.makedirs(cropped_save_dir, exist_ok=True)
    os.makedirs(patch_save_dir, exist_ok=True)
    logging.info(f"\n[process_and_save_input_image_patches] Loading input: {input_image_path}")
    try:
        image = np.array(Image.open(input_image_path).convert("RGB"))
    except Exception as e:
        logging.error(f"[process_and_save_input_image_patches] Failed to open {input_image_path}: {e}")
        return

    for i, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        bbox_patch = image[y_min:y_max, x_min:x_max]

        try:
            center = calculate_center_of_mass(bbox_patch)
        except Exception as e:
            logging.error(f"[process_and_save_input_image_patches] Center-of-mass error for box #{i}: {e}")
            continue

        center_global = (center[0] + x_min, center[1] + y_min)
        try:
            cropped_patch_pil = Image.fromarray(bbox_patch)
            crop_filename = f"{os.path.splitext(original_filename)[0]}_crop{i}"
            cropped_patch_path = os.path.join(cropped_save_dir, f"{crop_filename}.png")
            cropped_patch_pil.save(cropped_patch_path)
            logging.info(f"[INFO] Cropped object saved at: {cropped_patch_path} (Box #{i})")
        except Exception as e:
            logging.error(f"[ERROR] Failed to save crop for box #{i}: {e}")
            continue

        try:
            patch = extract_center_patch(image, center_global, patch_size=patch_size)
            patch_pil = Image.fromarray(patch)
            patch_filename = f"{os.path.splitext(original_filename)[0]}_patch{i}"
            patch_path = os.path.join(patch_save_dir, f"{patch_filename}.png")
            patch_pil.save(patch_path)
            logging.info(f"[INFO] Center-based patch saved: {patch_path} (Box #{i})")
        except ValueError as ve:
            logging.warning(f"[WARNING] {ve} => skipping patch for box #{i}.")
        except Exception as e:
            logging.error(f"[ERROR] Failed to save center patch for box #{i}: {e}")

##############################################################################
# MAIN PIPELINE
##############################################################################
def main_pipeline(
    input_image_path: str,
    vectorstore: FAISS,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    model: torch.nn.Module,
    mask_predictor: SamPredictor,
    sam_device: torch.device,
    chat: ChatOpenAI,
    use_lab: bool = False,
    patch_size: tuple = (100, 100),
    label_to_id_map: dict = {},
    target_objects: list = [],
    output_folder: str = "output",
    ripeness_prompt: str = ""
):
    processed_images_dir = os.path.join(output_folder, "processed_images")
    images_dir = os.path.join(output_folder, "images")
    labels_vdb_dir = os.path.join(output_folder, "labels_vdb_bbox")
    labels_orchestrator_dir = os.path.join(output_folder, "labels_gpt_bbox")
    labels_vdb_polygon_dir = os.path.join(output_folder, "labels_vdb_polygon")
    labels_orchestrator_polygon_dir = os.path.join(output_folder, "labels_gpt_polygon")
    cropped_dir = os.path.join(output_folder, "cropped")
    patch_dir = os.path.join(output_folder, "patches")

    for d in [
        processed_images_dir,
        images_dir,
        labels_vdb_dir,
        labels_orchestrator_dir,
        labels_vdb_polygon_dir,
        labels_orchestrator_polygon_dir,
        cropped_dir,
        patch_dir
    ]:
        os.makedirs(d, exist_ok=True)

    logging.info(f"\n[main_pipeline] Segmenting objects in: {input_image_path}")
    try:
        image_source, image_transformed = load_image(input_image_path)
    except Exception as e:
        logging.error(f"[main_pipeline] Failed to load {input_image_path}: {e}")
        return

    TEXT_PROMPT = " ".join(target_objects)
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    try:
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
    except Exception as e:
        logging.error(f"[main_pipeline] GroundingDINO predict error: {e}")
        return

    logging.info(f"[main_pipeline] Detected {len(boxes)} bounding boxes.")
    for i, box in enumerate(boxes):
        logging.info(f"Box {i}: {box}")

    H_trans, W_trans = image_transformed.shape[1:3]
    H_orig, W_orig, _ = image_source.shape

    rescaled_boxes = []
    for bbox_torch in boxes:
        x_center_norm, y_center_norm, width_norm, height_norm = bbox_torch.tolist()
        x_min_abs = int((x_center_norm - width_norm / 2) * W_orig)
        y_min_abs = int((y_center_norm - height_norm / 2) * H_orig)
        x_max_abs = int((x_center_norm + width_norm / 2) * W_orig)
        y_max_abs = int((y_center_norm + height_norm / 2) * H_orig)
        validated_bbox = validate_bbox([x_min_abs, y_min_abs, x_max_abs, y_max_abs],
                                       W_orig, H_orig)
        rescaled_boxes.append(validated_bbox)

    logging.info(f"\n[main_pipeline] Saving patches of detected objects...")
    original_filename = os.path.basename(input_image_path)
    process_and_save_input_image_patches(
        input_image_path=input_image_path,
        boxes=rescaled_boxes,
        original_filename=original_filename,
        cropped_save_dir=cropped_dir,
        patch_save_dir=patch_dir,
        patch_size=patch_size
    )

    try:
        shutil.copy(input_image_path, os.path.join(images_dir, original_filename))
        logging.info(f"[main_pipeline] Copied original: {os.path.join(images_dir, original_filename)}")
    except Exception as e:
        logging.error(f"[main_pipeline] Failed to copy original: {e}")

    logging.info(f"\n[main_pipeline] Classifying each segmented region with two-step GPT if available...")
    results = classify_segmented_objects(
        image_rgb=image_source,
        boxes=rescaled_boxes,
        clip_model=clip_model,
        processor=processor,
        device=device,
        vectorstore=vectorstore,
        chat=chat,
        top_k=1,
        use_lab=use_lab,
        cropped_save_dir=cropped_dir,
        patch_save_dir=patch_dir,
        patch_size=patch_size,
        label_to_id_map=label_to_id_map,
        target_objects=target_objects,
        ripeness_prompt=ripeness_prompt
    )

    logging.info(f"\n[main_pipeline] Annotating image + polygons...")
    annotate_image(
        image_rgb=image_source,
        results=results,
        output_image_path=os.path.join(processed_images_dir, original_filename),
        labels_vdb_dir=labels_vdb_dir,
        labels_orchestrator_dir=labels_orchestrator_dir,
        labels_vdb_polygon_dir=labels_vdb_polygon_dir,
        labels_orchestrator_polygon_dir=labels_orchestrator_polygon_dir,
        W_orig=W_orig,
        H_orig=H_orig,
        mask_predictor=mask_predictor,
        device=sam_device,
        label_to_id_map=label_to_id_map
    )

##############################################################################
# Helper Functions
##############################################################################
def validate_bbox(bbox: list, image_width: int, image_height: int) -> list:
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(x_min, image_width - 1))
    y_min = max(0, min(y_min, image_height - 1))
    x_max = max(0, min(x_max, image_width - 1))
    y_max = max(0, min(y_max, image_height - 1))
    return [x_min, y_min, x_max, y_max]

def get_class_id(label: str, label_to_id_map: dict) -> int:
    return label_to_id_map.get(label, len(label_to_id_map))

def determine_label_from_config(config_path: str) -> list:
    reference_images = []
    try:
        with open(config_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_filename = row['image_filename'].strip()
                label = row['label'].strip()
                reference_images.append((image_filename, label))
    except Exception as e:
        logging.error(f"[determine_label_from_config] Failed to read {config_path}: {e}")
    return reference_images

##############################################################################
# Argument Parser
##############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Ripeness Assessment Script")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--reference_images_folder", type=str, required=True)
    parser.add_argument("--input_images_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--groundingdino_config", type=str, required=True)
    parser.add_argument("--groundingdino_weights", type=str, required=True)
    parser.add_argument("--clip_model_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--use_lab", action='store_true')
    parser.add_argument("--patch_width", type=int, required=True)
    parser.add_argument("--patch_height", type=int, required=True)
    parser.add_argument("--gpt_model", type=str, default="gpt-4")
    parser.add_argument("--target_objects", nargs='+', required=True)
    parser.add_argument("--persistent", action='store_true')
    return parser.parse_args()

##############################################################################
# MAIN EXECUTION
##############################################################################
def main():
    args = parse_arguments()

    reference_images = determine_label_from_config(args.config_file)
    if not reference_images:
        logging.error("[ERROR] No reference images found in config.")
        exit(1)

    reference_images = [(os.path.join(args.reference_images_folder, img), label)
                        for img, label in reference_images]
    unique_labels = sorted(list(set([label for _, label in reference_images])))
    label_to_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    logging.info(f"[Main] label_to_id_map: {label_to_id_map}")

    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            try:
                with open(args.prompt_file, "r", encoding='utf-8') as f:
                    custom_prompt = f.read()
                logging.info(f"[Main] Loaded custom prompt from {args.prompt_file}")
            except Exception as e:
                logging.error(f"[ERROR] read prompt file {args.prompt_file}: {e}")
                custom_prompt = " ".join(args.target_objects)
        else:
            logging.error(f"[ERROR] Prompt file not found: {args.prompt_file}")
            custom_prompt = " ".join(args.target_objects)
    else:
        custom_prompt = " ".join(args.target_objects)

    patch_size = (args.patch_width, args.patch_height)
    logging.info(f"[Main] patch_size: {patch_size}")

    logging.info("\n[Main] Loading GroundingDINO model...")
    try:
        model = load_groundingdino_model(args.groundingdino_config,
                                         args.groundingdino_weights)
    except Exception as e:
        logging.error(f"[Main] Failed to load GroundingDINO: {e}")
        exit(1)

    logging.info("\n[Main] Loading CLIP model...")
    try:
        clip_model, processor, device = load_clip_model(args.clip_model_dir,
                                                        device=None)
    except Exception as e:
        logging.error(f"[Main] Failed to load CLIP: {e}")
        exit(1)

    vector_db_dir = os.path.join(args.output_folder, "vector_db")

    logging.info("\n[Main] Building or loading vector store...")
    try:
        vectorstore = build_reference_vectorstore(
            reference_images=reference_images,
            clip_model=clip_model,
            processor=processor,
            device=device,
            use_lab=args.use_lab,
            patch_save_dir=vector_db_dir,
            patch_size=patch_size,
            persistent=args.persistent,
            vector_db_dir=vector_db_dir
        )
    except Exception as e:
        logging.error(f"[Main] Failed to build vector store: {e}")
        exit(1)

    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
        logging.info("\n[Main] Initializing ChatOpenAI model...")
        try:
            chat = ChatOpenAI(model_name=args.gpt_model, max_tokens=1000)
            logging.info("[Main] ChatOpenAI ready.")
        except Exception as e:
            logging.error(f"[Main] ChatOpenAI init error: {e}")
            chat = None
    else:
        chat = None
        logging.info("\n[Main] No OpenAI key => skipping GPT-based steps.")

    logging.info("\n[Main] Loading SAM model for polygon generation...")
    try:
        mask_predictor, sam_device = load_sam_model(model_type="vit_h",
                                                    checkpoint_path="sam_vit_h.pth")
    except Exception as e:
        logging.error(f"[Main] SAM load error: {e}")
        exit(1)

    for root, dirs, files in os.walk(args.input_images_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_image_path = os.path.join(root, file)
                logging.info(f"\n[Main] Processing: {input_image_path}")
                main_pipeline(
                    input_image_path=input_image_path,
                    vectorstore=vectorstore,
                    clip_model=clip_model,
                    processor=processor,
                    device=device,
                    model=model,
                    mask_predictor=mask_predictor,
                    sam_device=sam_device,
                    chat=chat,  # if None => no GPT calls
                    use_lab=args.use_lab,
                    patch_size=patch_size,
                    label_to_id_map=label_to_id_map,
                    target_objects=args.target_objects,
                    output_folder=args.output_folder,
                    ripeness_prompt=custom_prompt
                )

if __name__ == "__main__":
    main()
