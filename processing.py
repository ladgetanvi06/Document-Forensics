"""
Document Forensics: Suspicious Edit Detection in Scanned Documents
------------------------------------------------------------------
Core Image Processing Pipeline Module.
Contains beginner-friendly, modular OpenCV functions for comparing
two document images and highlighting detected discrepancies.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, Any, Tuple, Optional, List


def load_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes an uploaded image byte buffer into a standard BGR NumPy array (OpenCV format).
    Returns None if the image cannot be decoded or is empty.
    """
    if not file_bytes or len(file_bytes) == 0:
        return None
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def encode_image_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    """
    Encodes a BGR image array into PNG format bytes for file downloading or streaming.
    """
    if img_bgr is None or img_bgr.size == 0:
        return b""
    success, encoded_img = cv2.imencode(".png", img_bgr)
    if success:
        return encoded_img.tobytes()
    return b""


def load_image_from_path(filepath: str) -> Optional[np.ndarray]:
    """
    Loads an image from local disk into a BGR NumPy array.
    """
    try:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def check_and_match_dimensions(
    img_orig: np.ndarray, 
    img_susp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Validates whether the two images have identical spatial dimensions (height, width).
    If dimensions differ:
      - Resizes the suspected image to match the original image dimensions.
      - Returns a warning flag and explanatory message disclosing that resizing may
        introduce interpolation artifacts or alignment shifts.
    """
    h_orig, w_orig = img_orig.shape[:2]
    h_susp, w_susp = img_susp.shape[:2]

    if (h_orig, w_orig) == (h_susp, w_susp):
        return img_orig, img_susp, False, "Dimensions match perfectly."

    warning_msg = (
        f"Dimension mismatch detected! Original: ({w_orig}x{h_orig}), "
        f"Suspected: ({w_susp}x{h_susp}). "
        f"The suspected document was automatically resized to ({w_orig}x{h_orig}) "
        f"for comparison. Note: Resizing or slight scanning misalignments can cause "
        f"false positives along document margins."
    )
    
    # Resize suspected image to match original image dimensions using bilinear interpolation
    img_susp_resized = cv2.resize(img_susp, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return img_orig, img_susp_resized, True, warning_msg


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """
    Converts a 3-channel BGR color image to a 1-channel Grayscale image.
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    Simplifies pixel comparison by reducing 3 color dimensions to single intensity values.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_spatial_filter(
    gray_image: np.ndarray, 
    filter_type: str = "gaussian", 
    kernel_size: int = 5
) -> np.ndarray:
    """
    Applies spatial smoothing to reduce high-frequency scanner noise, grain, and paper texture.
    - Gaussian Filter: Weighted average using Gaussian distribution; preserves edges reasonably well.
    - Median Filter: Replaces center pixel with median of neighborhood; excellent for salt-and-pepper scan noise.
    """
    # Ensure kernel_size is an odd positive integer
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(1, kernel_size)

    if kernel_size == 1:
        return gray_image.copy()

    if filter_type.lower() == "median":
        return cv2.medianBlur(gray_image, kernel_size)
    else:
        # Default: Gaussian Blur
        return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def compute_absolute_difference(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """
    Calculates pixel-by-pixel absolute difference: Diff(x, y) = |Gray1(x, y) - Gray2(x, y)|.
    Unchanged pixels yield values close to 0 (black).
    Modified, added, or removed text regions yield higher values (white / bright gray).
    """
    return cv2.absdiff(gray1, gray2)


def apply_thresholding(
    diff_image: np.ndarray, 
    threshold_val: int = 30, 
    use_otsu: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Converts grayscale difference map into a pure binary mask (0 or 255).
    - Manual threshold: Pixels > threshold_val become 255 (changed), others 0 (background).
    - Otsu's threshold: Automatically calculates optimal threshold separating foreground from background.
    """
    if use_otsu:
        actual_thresh, thresh_img = cv2.threshold(
            diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        actual_thresh = float(threshold_val)
        _, thresh_img = cv2.threshold(
            diff_image, threshold_val, 255, cv2.THRESH_BINARY
        )
    return thresh_img, actual_thresh


def apply_morphological_cleanup(
    binary_mask: np.ndarray, 
    kernel_size: int = 3, 
    iterations: int = 2
) -> np.ndarray:
    """
    Applies morphological operations (Dilation followed by Closing) to:
    1. Fill gaps inside individual text characters.
    2. Bridge fragmented stroke differences into cohesive bounding blobs.
    3. Remove isolated single-pixel scanner speckles (noise reduction).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Dilate to connect nearby character parts
    dilated = cv2.dilate(binary_mask, kernel, iterations=iterations)
    # Morphological Close (Dilation then Erosion) to fill internal holes
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def detect_and_highlight_changes(
    clean_mask: np.ndarray, 
    suspected_bgr: np.ndarray, 
    min_area: int = 80
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Finds contours from the cleaned binary mask and filters them by minimum area.
    Draws highlighted bounding boxes and index numbers on a copy of the suspected image.
    
    Returns:
      - Annotated BGR image with distinct red bounding boxes and badge tags.
      - List of detected region dictionaries containing (x, y, w, h, area).
    """
    # cv2.findContours finds connected boundaries in binary image
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = suspected_bgr.copy()
    detected_regions: List[Dict[str, Any]] = []

    region_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            detected_regions.append({
                "id": region_id,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": float(area)
            })

            # Draw a solid rectangular bounding box around the suspect area
            # Color: Bright Red (0, 0, 255 in BGR)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 230), 2)

            # Draw a small label badge with the region ID
            label = f"#{region_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # Badge background
            badge_y1 = max(0, y - text_h - 6)
            badge_y2 = y
            cv2.rectangle(annotated, (x, badge_y1), (x + text_w + 6, badge_y2), (0, 0, 230), -1)
            # Badge text (White)
            cv2.putText(
                annotated, 
                label, 
                (x + 3, y - 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )

            region_id += 1

    return annotated, detected_regions


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Utility to convert OpenCV BGR format to RGB for Streamlit/PIL rendering."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run_document_forensics_pipeline(
    img_orig_bgr: np.ndarray,
    img_susp_bgr: np.ndarray,
    filter_type: str = "gaussian",
    kernel_size: int = 3,
    threshold_val: int = 30,
    use_otsu: bool = False,
    morph_kernel: int = 3,
    morph_iterations: int = 2,
    min_area: int = 40
) -> Dict[str, Any]:
    """
    Executes the end-to-end document forensic comparison pipeline:
    1. Dimension check & resize alignment
    2. Grayscale conversion
    3. Noise filtering (Gaussian or Median)
    4. Absolute difference calculation
    5. Binary thresholding (Manual or Otsu)
    6. Morphological cleanup (dilation & closing)
    7. Contour detection & bounding box highlighting

    Returns a comprehensive dictionary containing all intermediate visual outputs,
    metadata, and detected discrepancy regions.
    """
    # Step 1: Align & match dimensions
    orig_aligned, susp_aligned, resized, warning_msg = check_and_match_dimensions(
        img_orig_bgr, img_susp_bgr
    )

    # Step 2: Convert to Grayscale
    gray_orig = to_grayscale(orig_aligned)
    gray_susp = to_grayscale(susp_aligned)

    # Step 3: Spatial Filtering
    filtered_orig = apply_spatial_filter(gray_orig, filter_type, kernel_size)
    filtered_susp = apply_spatial_filter(gray_susp, filter_type, kernel_size)

    # Step 4: Absolute Difference
    diff_map = compute_absolute_difference(filtered_orig, filtered_susp)

    # Step 5: Thresholding
    binary_mask, actual_thresh = apply_thresholding(diff_map, threshold_val, use_otsu)

    # Step 6: Morphological Cleanup
    clean_mask = apply_morphological_cleanup(binary_mask, morph_kernel, morph_iterations)

    # Step 7: Contour Detection and Highlight
    highlighted_susp, detected_regions = detect_and_highlight_changes(
        clean_mask, susp_aligned, min_area
    )

    return {
        "resized_warning": warning_msg if resized else None,
        "is_resized": resized,
        "dimensions": (orig_aligned.shape[1], orig_aligned.shape[0]),
        "orig_aligned_rgb": bgr_to_rgb(orig_aligned),
        "susp_aligned_rgb": bgr_to_rgb(susp_aligned),
        "gray_orig": gray_orig,
        "gray_susp": gray_susp,
        "filtered_orig": filtered_orig,
        "filtered_susp": filtered_susp,
        "diff_map": diff_map,
        "binary_mask": binary_mask,
        "clean_mask": clean_mask,
        "highlighted_susp_rgb": bgr_to_rgb(highlighted_susp),
        "highlighted_susp_bgr": highlighted_susp,
        "detected_regions": detected_regions,
        "total_regions": len(detected_regions),
        "applied_threshold": actual_thresh,
    }
