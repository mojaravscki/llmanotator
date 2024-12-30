import os
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

##############################################################################
# Utility Functions
##############################################################################
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
        print(f"[ERROR] Empty patch extracted at center: {center}")
        raise ValueError("Empty patch extracted. Check center coordinates.")

    return patch


def extract_and_save_patch(image: np.ndarray, center: tuple, save_path: str, patch_name: str, patch_size=(100, 100)):
    """Extract a patch centered on the center of mass, save it to disk, and return the patch."""
    patch = extract_center_patch(image, center, patch_size)
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
    save_file = os.path.join(save_path, f"{patch_name}.png")
    cv2.imwrite(save_file, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))  # Save patch as PNG (convert RGB to BGR for OpenCV)
    print(f"[INFO] Patch saved at: {save_file}")
    return patch


##############################################################################
# Extract CLIP Embeddings
##############################################################################
def get_image_embedding(image: Image.Image, clip_model, processor, device, use_lab=False):
    """Get normalized CLIP embedding for a PIL image."""
    if use_lab:
        np_image = np.array(image)
        np_image_lab = convert_to_lab(np_image)
        image = Image.fromarray(np_image_lab)

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.squeeze(0).cpu().numpy()
    return embedding


##############################################################################
# GroundingDINO Loader
##############################################################################
def load_groundingdino_model(config_path, weights_path, device=None):
    """Load the GroundingDINO model."""
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[load_groundingdino_model] Using device: {device}")

    model = load_model(config_path, weights_path)
    model = model.to(device)
    return model

##############################################################################
# CLIP Loader
##############################################################################
def load_clip_model(device=None):
    """Load the local CLIP model."""
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[load_clip_model] Using device: {device}")

    LOCAL_CLIP_DIR = "/Users/macbookpro/Documents/Projects/kws/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(LOCAL_CLIP_DIR).to(device)
    processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_DIR)
    return clip_model, processor, device

##############################################################################
# Build Reference Vector Store
##############################################################################
def build_reference_vectorstore(reference_images, clip_model, processor, device, use_lab=False, patch_save_dir="patches/vector_db") -> FAISS:
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
            print(f"[WARNING] File not found: {img_path} — skipping.")
            continue

        print(f"\n[build_reference_vectorstore] Loading reference #{i}: {img_path}")
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
        print(f"[INFO] Original patch saved at: {original_patch_save_path}")

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
            print(f"[INFO] Augmented patch saved at: {aug_patch_save_path}")

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

    print(f"\n[build_reference_vectorstore] Building FAISS index from {len(text_embedding_tuples)} items.")
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_tuples,
        embedding=embedding_function,
        metadatas=all_metadatas,
    )
    return vectorstore

##############################################################################
# Process Input Image Patches
##############################################################################
def process_and_save_input_image_patches(input_image_path, boxes, patch_save_dir="patches/input_image", patch_size=(100, 100)):
    """
    Extract and save patches from the input image based on detected bounding boxes' centers of mass.
    """
    os.makedirs(patch_save_dir, exist_ok=True)
    print(f"\n[process_and_save_input_image_patches] Loading input image: {input_image_path}")
    image = np.array(Image.open(input_image_path).convert("RGB"))

    for i, bbox in enumerate(boxes):
        # Calculate center of mass for each bounding box
        x_min, y_min, x_max, y_max = map(int, bbox)
        bbox_patch = image[y_min:y_max, x_min:x_max]
        center = calculate_center_of_mass(bbox_patch)

        # Adjust center to the global coordinates
        center_global = (center[0] + x_min, center[1] + y_min)

        # Extract and save the patch
        extract_and_save_patch(image, center=center_global, save_path=patch_save_dir, patch_name=f"input_box_{i}", patch_size=patch_size)


##############################################################################
# CLASSIFY SEGMENTED OBJECTS
##############################################################################
def classify_segmented_objects(image_rgb, boxes, clip_model, processor, device, reference_vectorstore, top_k=1, use_lab=False):
    """Classify each segmented object by matching patches against the reference vectorstore."""
    results = []
    for j, bbox in enumerate(boxes):
        # Calculate center of mass for each bounding box
        x_min, y_min, x_max, y_max = map(int, bbox)
        bbox_patch = image_rgb[y_min:y_max, x_min:x_max]
        center = calculate_center_of_mass(bbox_patch)

        # Adjust center to the global coordinates
        center_global = (center[0] + x_min, center[1] + y_min)

        # Extract patch and generate embedding
        patch = extract_center_patch(image_rgb, center_global, patch_size=(100, 100))
        patch_pil = Image.fromarray(patch)
        emb = get_image_embedding(patch_pil, clip_model, processor, device, use_lab=use_lab)
        emb = emb.astype(np.float32).reshape(1, -1)

        # Perform FAISS search
        faiss_index = reference_vectorstore.index
        distances, indices = faiss_index.search(emb, top_k)

        matches = []
        for k in range(top_k):
            idx = indices[0][k]
            distance = distances[0][k]
            doc_id = reference_vectorstore.index_to_docstore_id[idx]
            doc = reference_vectorstore.docstore.search(doc_id)
            label = doc.metadata["label"]
            matches.append({"label": label, "distance": distance})
            print(f"Box #{j} - Match {k+1}: {label} with distance: {distance}")

        results.append({"bbox": bbox, "matches": matches})
    return results

##############################################################################
# Process and Save Detected Object Patches
##############################################################################
def process_and_save_input_image_patches(input_image_path, boxes, patch_save_dir="patches/detected_objects", patch_size=(100, 100)):
    """
    Extract and save patches from the input image based on detected bounding boxes' centers of mass.
    """
    os.makedirs(patch_save_dir, exist_ok=True)
    print(f"\n[process_and_save_input_image_patches] Loading input image: {input_image_path}")
    image = np.array(Image.open(input_image_path).convert("RGB"))

    for i, bbox in enumerate(boxes):
        # Calculate center of mass for each bounding box
        x_min, y_min, x_max, y_max = map(int, bbox)
        bbox_patch = image[y_min:y_max, x_min:x_max]
        center = calculate_center_of_mass(bbox_patch)

        # Adjust center to the global coordinates
        center_global = (center[0] + x_min, center[1] + y_min)

        # Extract and save the patch
        patch_save_path = os.path.join(patch_save_dir, f"detected_object_{i}")
        extract_and_save_patch(image, center=center_global, save_path=patch_save_path, patch_name=f"object_{i}", patch_size=patch_size)

##############################################################################
# MAIN PIPELINE
##############################################################################
def main_pipeline(input_image_path, reference_images, groundingdino_config, groundingdino_weights, use_lab=False):
    """
    Main pipeline for object detection, patch extraction, classification, and saving annotated image and patches.
    """
    print(f"\n[main_pipeline] Loading GroundingDINO model...")
    model = load_groundingdino_model(groundingdino_config, groundingdino_weights)

    print(f"\n[main_pipeline] Loading local CLIP model...")
    clip_model, processor, device = load_clip_model()

    print(f"\n[main_pipeline] Building vector store from reference images...")
    vectorstore = build_reference_vectorstore(reference_images, clip_model, processor, device, use_lab=use_lab)

    print(f"\n[main_pipeline] Segmenting objects in input image...")
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

    print(f"[main_pipeline] Detected {len(boxes)} bounding boxes.")
    for i, box in enumerate(boxes):
        print(f"Box {i}: {box}")

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
        rescaled_boxes.append([x_min_abs, y_min_abs, x_max_abs, y_max_abs])

    print(f"\n[main_pipeline] Saving patches of detected objects...")
    process_and_save_input_image_patches(
        input_image_path=input_image_path,
        boxes=rescaled_boxes,
        patch_save_dir="patches/detected_objects",
        patch_size=(100, 100)
    )

    print(f"\n[main_pipeline] Classifying each segmented region...")
    results = classify_segmented_objects(
        image_source,
        rescaled_boxes,
        clip_model,
        processor,
        device,
        reference_vectorstore=vectorstore,
        top_k=100,
        use_lab=use_lab
    )

    print(f"\n[main_pipeline] Annotating image with bounding boxes and labels...")
    annotated_frame = image_source.copy()
    for result, rescaled_bbox in zip(results, rescaled_boxes):
        x_min, y_min, x_max, y_max = rescaled_bbox
        label = result["matches"][0]["label"]

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Draw label
        cv2.putText(
            annotated_frame,
            label,
            (x_min, max(y_min - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Save the annotated image
    output_path = "output_annotated_with_classes.jpg"
    cv2.imwrite(output_path, annotated_frame[..., ::-1])  # Convert RGB to BGR for saving
    print(f"[main_pipeline] Annotated image with classes saved at: {output_path}")


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
    #input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/valid/images/APC_0346_jpg.rf.dda372424b2966557e9fed6d8fe70006.jpg"
    #input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3256_jpeg.rf.882cfdb438039a18c8bca512c93e30f9.jpg"
    #input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3190_jpeg.rf.44d0ee7fbb76c87f49247a0815e6ec2b.jpg"
    input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3188_jpeg.rf.ac483475adf281092607e95216a50140.jpg"
    input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3183_jpeg.rf.8818a3d884b56d0bf19ad90311649a32.jpg"
    #input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3174_jpeg.rf.e34bef1792ba29ca923cbbfe3a895386.jpg"
    #input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3160_jpeg.rf.fe9c16120c44f10b1c1a01b7c40f75c8.jpg"
    #input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3146_jpeg.rf.b53748504038741eb5baf5457391ad65.jpg"
    
    #input_image_path = "/Users/macbookpro/Documents/Projects/phd/roboflow/no_aug/h1_asis/consolidado_multiclasse-4/train/images/IMG_3111_jpeg.rf.b4156c5076b9dd9827809c01ffcca46a.jpg"
    # Run the pipeline
    main_pipeline(
        input_image_path=input_image_path,
        reference_images=reference_images,
        groundingdino_config=groundingdino_config,
        groundingdino_weights=groundingdino_weights,
        use_lab=True
    )
