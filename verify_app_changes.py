"""
verify_app_changes.py
---------------------
Script to verify that app.py complies with all user requirements:
1. "Step-by-Step Image Processing Pipeline (Viva Explanation)" and tabs removed
2. Prototype badge removed
3. Deploy button hidden via CSS & config.toml
4. Quick Test Samples dropdown removed
5. Both document uploaders present
6. Document comparison and download generation work cleanly
"""

import os
import re
from processing import (
    load_image_from_path,
    run_document_forensics_pipeline,
    encode_image_to_png_bytes
)


def verify_codebase():
    print("=" * 60)
    print("VERIFYING APP.PY UI & ARCHITECTURE CHANGES")
    print("=" * 60)

    with open("app.py", "r", encoding="utf-8") as f:
        app_code = f.read()

    # 1. Check removals
    assert "Step-by-Step Image Processing Pipeline" not in app_code, \
        "FAILED: Pipeline explanation section was not removed!"
    assert "Viva Explanation" not in app_code, \
        "FAILED: Viva explanation label was not removed!"
    assert "pipeline_tabs" not in app_code, \
        "FAILED: Pipeline tabs were not removed!"
    assert "badge-sip" not in app_code, \
        "FAILED: Prototype badge CSS was not removed!"
    assert "SIP Image Processing Project Prototype" not in app_code, \
        "FAILED: Prototype badge text was not removed!"
    assert "Quick Test Samples" not in app_code, \
        "FAILED: Quick Test Samples dropdown was not removed!"
    assert "sample_options" not in app_code, \
        "FAILED: sample_options dict was not removed!"

    print("[PASS] 1. Explanation section, tabs, prototype badge, and sample dropdown successfully removed.")

    # 2. Check title and subtitle preserved
    assert "Document Forensics: Suspicious Edit Detection" in app_code, \
        "FAILED: Main title missing!"
    assert "Compare an authentic reference document with a suspected version" in app_code, \
        "FAILED: Professional subtitle missing!"
    print("[PASS] 2. Main title and professional subtitle preserved.")

    # 3. Check deploy button hidden
    assert ".stDeployButton" in app_code and "display: none !important" in app_code, \
        "FAILED: Deploy button CSS rule missing!"
    assert os.path.exists(".streamlit/config.toml"), \
        "FAILED: .streamlit/config.toml missing!"
    with open(".streamlit/config.toml", "r", encoding="utf-8") as f:
        config_text = f.read()
    assert 'toolbarMode = "minimal"' in config_text or 'toolbarMode = "viewer"' in config_text, \
        "FAILED: toolbarMode setting missing in config.toml!"
    print("[PASS] 3. Deploy button and toolbar controls hidden via CSS and Streamlit config.")

    # 4. Check uploaders & upload-only workflow
    assert 'orig_file = st.sidebar.file_uploader(' in app_code, \
        "FAILED: Original file uploader missing!"
    assert 'susp_file = st.sidebar.file_uploader(' in app_code, \
        "FAILED: Suspected file uploader missing!"
    assert 'both_uploaded = (orig_file is not None) and (susp_file is not None)' in app_code, \
        "FAILED: both_uploaded check missing!"
    print("[PASS] 4. User-only upload controls and dual-document availability check verified.")

    # 5. Check side-by-side comparison & download button
    assert "Side-by-Side Document Inspection" in app_code, \
        "FAILED: Side-by-side inspection header missing!"
    assert "st.download_button" in app_code, \
        "FAILED: Download button missing!"
    assert "encode_image_to_png_bytes" in app_code, \
        "FAILED: PNG byte encoding missing!"
    assert "Detected Regions Inventory" in app_code, \
        "FAILED: Detected regions inventory table missing!"
    print("[PASS] 5. Side-by-side inspection, inventory table, and download features intact.")

    # 6. Functional test of comparison pipeline on images
    orig = load_image_from_path("samples/pair2_amount_orig.png")
    susp = load_image_from_path("samples/pair2_amount_tampered.png")
    res = run_document_forensics_pipeline(orig, susp)
    assert res["total_regions"] >= 1, "Pipeline failed to detect discrepancies!"
    png_bytes = encode_image_to_png_bytes(res["highlighted_susp_bgr"])
    assert len(png_bytes) > 1000, "Failed to encode download bytes!"
    print(f"[PASS] 6. Functional test: Detected {res['total_regions']} regions, generated {len(png_bytes)} bytes of downloadable PNG.")

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    verify_codebase()
