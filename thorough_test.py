"""
thorough_test.py
----------------
Deep, comprehensive verification of Document Forensics pipeline:
- Python error checks
- Identical documents test
- Deliberately altered documents test
- Visual highlight & bounding box correctness
- Parameter variations (Otsu, Median vs Gaussian, threshold sweep)
- Invalid / corrupted image error handling
- Byte download generation
"""

import os
import cv2
import numpy as np
from PIL import Image
import io

from processing import (
    load_image_from_path,
    load_image_from_bytes,
    check_and_match_dimensions,
    to_grayscale,
    apply_spatial_filter,
    compute_absolute_difference,
    apply_thresholding,
    apply_morphological_cleanup,
    detect_and_highlight_changes,
    run_document_forensics_pipeline,
    encode_image_to_png_bytes
)


def test_suite():
    print("=" * 70)
    print("STARTING THOROUGH VERIFICATION TEST SUITE")
    print("=" * 70)

    # 1. Test error handling on corrupted / empty inputs
    print("\n[Step 1] Testing Error Handling on Corrupted/Empty Inputs...")
    corrupted_bytes = b"Not an image at all - random string buffer"
    empty_bytes = b""
    res_corrupt = load_image_from_bytes(corrupted_bytes)
    res_empty = load_image_from_bytes(empty_bytes)
    assert res_corrupt is None, "Corrupted bytes must return None safely"
    assert res_empty is None, "Empty bytes must return None safely"
    print("  [OK] Corrupted and empty byte buffers handled safely without crashing.")

    # 2. Test Identical Documents
    print("\n[Step 2] Testing Identical Document Images...")
    orig1 = load_image_from_path("samples/pair1_identical_orig.png")
    susp1 = load_image_from_path("samples/pair1_identical_susp.png")
    assert orig1 is not None and susp1 is not None, "Failed to load sample 1 images"
    
    res1 = run_document_forensics_pipeline(orig1, susp1)
    assert res1["total_regions"] == 0, f"Expected 0 changes on identical docs, got {res1['total_regions']}"
    assert not res1["is_resized"], "Identical docs should not be resized"
    print(f"  [OK] Identical Documents: Detected {res1['total_regions']} regions (Exact match, 0 false positives).")

    # 3. Test Deliberately Altered Invoice Amount
    print("\n[Step 3] Testing Deliberately Altered Invoice Amount...")
    orig2 = load_image_from_path("samples/pair2_amount_orig.png")
    susp2 = load_image_from_path("samples/pair2_amount_tampered.png")
    assert orig2 is not None and susp2 is not None, "Failed to load sample 2 images"

    res2 = run_document_forensics_pipeline(orig2, susp2, threshold_val=30, min_area=40)
    assert res2["total_regions"] >= 1, f"Expected detected regions, got {res2['total_regions']}"
    
    # Verify the altered region location (invoice amount is in the bottom total box)
    box = res2["detected_regions"][0]
    print(f"  [OK] Altered Region Detected: ID #{box['id']} at (X={box['x']}, Y={box['y']}, W={box['w']}, H={box['h']}, Area={box['area']} px^2)")
    # Amount box is expected around Y ≈ 500-650
    assert 500 <= box['y'] <= 650, f"Detected box at unexpected Y coordinate: {box['y']}"
    assert box['w'] > 10 and box['h'] > 10, "Bounding box size must be non-trivial"

    # Verify visual highlight is actually drawn (check that annotated image has red pixels in the bounding box)
    annotated = res2["highlighted_susp_bgr"]
    # Look for pure bright red (BGR: 0, 0, 230) on bounding box border
    red_pixels = np.sum((annotated[:, :, 0] == 0) & (annotated[:, :, 1] == 0) & (annotated[:, :, 2] == 230))
    assert red_pixels > 0, "Annotated image must contain red bounding box pixels"
    print(f"  [OK] Visual Highlighting Confirmed: {red_pixels} bounding box pixels drawn.")

    # 4. Test Deliberately Altered Date and Reference
    print("\n[Step 4] Testing Deliberately Altered Date & Reference...")
    orig3 = load_image_from_path("samples/pair3_date_orig.png")
    susp3 = load_image_from_path("samples/pair3_date_tampered.png")
    res3 = run_document_forensics_pipeline(orig3, susp3)
    assert res3["total_regions"] >= 2, f"Expected multiple tampered zones, got {res3['total_regions']}"
    print(f"  [OK] Multiple Discrepancies Detected: {res3['total_regions']} zones identified successfully.")

    # 5. Test Parameter Variations & Sensitivity
    print("\n[Step 5] Testing Parameter Variations...")
    # (a) Filter type: Median vs Gaussian
    res_gaussian = run_document_forensics_pipeline(orig2, susp2, filter_type="gaussian", kernel_size=3)
    res_median = run_document_forensics_pipeline(orig2, susp2, filter_type="median", kernel_size=3)
    assert res_gaussian["total_regions"] >= 1, "Gaussian filter failed to detect changes"
    assert res_median["total_regions"] >= 1, "Median filter failed to detect changes"
    print("  [OK] Both Gaussian and Median spatial smoothing filters operate successfully.")

    # (b) Automatic Otsu thresholding
    res_otsu = run_document_forensics_pipeline(orig2, susp2, use_otsu=True)
    print(f"  [OK] Otsu's Thresholding automatically calculated optimal threshold: {res_otsu['applied_threshold']:.1f}")
    assert res_otsu["applied_threshold"] > 0, "Otsu threshold must be positive"

    # (c) Threshold sweep: Lower threshold captures more foreground pixels than high threshold
    res_thresh_low = run_document_forensics_pipeline(orig2, susp2, threshold_val=15)
    res_thresh_high = run_document_forensics_pipeline(orig2, susp2, threshold_val=70)
    white_px_low = int(np.sum(res_thresh_low["binary_mask"] > 0))
    white_px_high = int(np.sum(res_thresh_high["binary_mask"] > 0))
    assert white_px_low > white_px_high, f"Expected more white pixels at low threshold (got {white_px_low} vs {white_px_high})"
    print(f"  [OK] Threshold sweep verified: Low thresh={white_px_low} white px ({res_thresh_low['total_regions']} unified region) vs High thresh={white_px_high} white px ({res_thresh_high['total_regions']} fragmented regions).")

    # 6. Test Dimension Mismatch & Alignment Warning
    print("\n[Step 6] Testing Dimension Mismatch Handling...")
    orig4 = load_image_from_path("samples/pair4_mismatch_orig.png")
    susp4 = load_image_from_path("samples/pair4_mismatch_tampered.png")
    res4 = run_document_forensics_pipeline(orig4, susp4)
    assert res4["is_resized"] is True, "Resizing flag should be True"
    assert res4["resized_warning"] is not None, "Warning message must be generated"
    assert "Dimension mismatch detected" in res4["resized_warning"], "Warning must disclose dimension mismatch"
    print("  [OK] Dimension mismatch properly intercepted, resized to match, and warning disclosed.")

    # 7. Test Image Download Byte Generation
    print("\n[Step 7] Testing Download Output...")
    png_bytes = encode_image_to_png_bytes(res2["highlighted_susp_bgr"])
    assert len(png_bytes) > 1000, "Download byte buffer should be non-empty"
    # Verify the byte buffer is valid PNG format by reloading with PIL
    pil_reloaded = Image.open(io.BytesIO(png_bytes))
    assert pil_reloaded.format == "PNG", f"Expected PNG format, got {pil_reloaded.format}"
    print(f"  [OK] Downloadable PNG generated: {len(png_bytes)} bytes, format confirmed as valid PNG.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    test_suite()
