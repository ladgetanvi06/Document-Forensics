"""
app.py
------
Document Forensics: Suspicious Edit Detection in Scanned Documents
A Professional Computer Vision Application using OpenCV and Streamlit.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io

from processing import (
    run_document_forensics_pipeline,
    load_image_from_bytes,
    encode_image_to_png_bytes,
    bgr_to_rgb
)

# Page Configuration
st.set_page_config(
    page_title="Document Forensics | Suspicious Edit Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS with High-Contrast Typography and Hidden Deploy Button
st.markdown("""
<style>
    /* Hide Streamlit Deploy button and standard branding */
    .stDeployButton, [data-testid="stDeployButton"], [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    #MainMenu { visibility: visible; }
    footer { visibility: hidden; }

    /* Main typography */
    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #94a3b8;
        margin-bottom: 1.6rem;
        line-height: 1.5;
    }

    /* Metric summary cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Status alert boxes */
    .alert-warning-box {
        background-color: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        color: #fbbf24;
        margin: 14px 0;
        font-size: 0.92rem;
    }
    .disclaimer-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 2.5rem;
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .disclaimer-box strong {
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header Section
    st.markdown('<div class="main-title">🔍 Document Forensics: Suspicious Edit Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Compare an authentic reference document with a suspected version to identify, highlight, '
        'and analyze local visual discrepancies using classical computer vision.'
        '</div>',
        unsafe_allow_html=True
    )

    # Sidebar: Document Inputs
    st.sidebar.header("📁 Document Inputs")

    orig_file = st.sidebar.file_uploader(
        "Upload Original (Reference) Document:",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="The authentic, unaltered reference document."
    )

    susp_file = st.sidebar.file_uploader(
        "Upload Suspected (Questioned) Document:",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="The document suspected of tampering, edit, or unauthorized revision."
    )

    # Sidebar: Pipeline Parameters
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Processing Parameters")

    filter_type = st.sidebar.radio(
        "Spatial Filter Type:",
        options=["Gaussian", "Median"],
        index=0,
        help="Gaussian: weighted averaging for gentle smoothing. Median: removes impulse/salt-and-pepper scan noise."
    )

    kernel_size = st.sidebar.select_slider(
        "Smoothing Kernel Size:",
        options=[1, 3, 5, 7, 9],
        value=3,
        help="Higher values suppress paper texture noise but may blur tiny font strokes."
    )

    use_otsu = st.sidebar.checkbox(
        "Use Otsu's Automatic Thresholding",
        value=False,
        help="Calculates threshold dynamically based on bimodal histogram."
    )

    threshold_val = st.sidebar.slider(
        "Difference Threshold (Intensity 0-255):",
        min_value=5,
        max_value=120,
        value=30,
        disabled=use_otsu,
        help="Minimum pixel absolute intensity change required to be marked as difference."
    )

    st.sidebar.markdown("##### Morphology & Filtering")
    morph_iterations = st.sidebar.slider(
        "Dilation Iterations:",
        min_value=1,
        max_value=5,
        value=2,
        help="Expands and connects individual stroke edits into unified word bounding boxes."
    )

    min_area = st.sidebar.slider(
        "Minimum Region Area (Pixels):",
        min_value=10,
        max_value=500,
        value=40,
        help="Contours smaller than this area threshold are ignored as insignificant scanner dust or noise."
    )

    # Verification of Document Availability
    both_uploaded = (orig_file is not None) and (susp_file is not None)

    compare_button = st.button(
        "🚀 Run Document Comparison",
        type="primary",
        use_container_width=True,
        disabled=not both_uploaded
    )

    if not both_uploaded:
        if orig_file is None and susp_file is None:
            st.info(
                "👈 **Ready to inspect:** Please upload both the **Original (Reference)** and "
                "**Suspected (Questioned)** document images from the sidebar to begin comparison."
            )
        elif orig_file is None:
            st.warning("⚠️ **Original Document Missing:** Please upload the authentic reference document in the sidebar.")
        elif susp_file is None:
            st.warning("⚠️ **Suspected Document Missing:** Please upload the questioned document in the sidebar.")
        return

    # Decode uploaded files
    orig_image_bgr = load_image_from_bytes(orig_file.read())
    susp_image_bgr = load_image_from_bytes(susp_file.read())

    if orig_image_bgr is None or susp_image_bgr is None or orig_image_bgr.size == 0 or susp_image_bgr.size == 0:
        st.error("❌ **Error reading images:** One or both files could not be decoded. Please verify file integrity.")
        return

    orig_name = orig_file.name
    susp_name = susp_file.name

    # Execute Document Forensics Pipeline
    with st.spinner("Processing document comparison..."):
        results = run_document_forensics_pipeline(
            img_orig_bgr=orig_image_bgr,
            img_susp_bgr=susp_image_bgr,
            filter_type=filter_type.lower(),
            kernel_size=kernel_size,
            threshold_val=threshold_val,
            use_otsu=use_otsu,
            morph_kernel=3,
            morph_iterations=morph_iterations,
            min_area=min_area
        )

    # Dimension Mismatch Alert
    if results["is_resized"]:
        st.markdown(
            f'<div class="alert-warning-box">⚠️ <strong>Notice:</strong> {results["resized_warning"]}</div>',
            unsafe_allow_html=True
        )

    # Metrics Summary Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            f'<div class="metric-card"><h3>{results["total_regions"]}</h3><p>Suspicious Regions</p></div>',
            unsafe_allow_html=True
        )
    with col_m2:
        status_text = "Clean (Match)" if results["total_regions"] == 0 else "Discrepancy Found"
        status_color = "#22c55e" if results["total_regions"] == 0 else "#ef4444"
        st.markdown(
            f'<div class="metric-card"><h3 style="color: {status_color};">{status_text}</h3><p>Inspection Status</p></div>',
            unsafe_allow_html=True
        )
    with col_m3:
        w, h = results["dimensions"]
        st.markdown(
            f'<div class="metric-card"><h3>{w} × {h}</h3><p>Dimensions (px)</p></div>',
            unsafe_allow_html=True
        )
    with col_m4:
        st.markdown(
            f'<div class="metric-card"><h3>{results["applied_threshold"]:.0f}</h3><p>Applied Threshold</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Primary Side-by-Side Comparison View
    st.subheader("📊 Side-by-Side Document Inspection")
    col_orig, col_susp, col_highlight = st.columns(3)

    with col_orig:
        st.markdown(f"**Original (Reference):** `{orig_name}`")
        st.image(results["orig_aligned_rgb"], use_container_width=True)

    with col_susp:
        st.markdown(f"**Suspected Document:** `{susp_name}`")
        st.image(results["susp_aligned_rgb"], use_container_width=True)

    with col_highlight:
        st.markdown("**Forensic Inspection Result (Annotated):**")
        st.image(results["highlighted_susp_rgb"], use_container_width=True)

    # Download Button
    download_bytes = encode_image_to_png_bytes(results["highlighted_susp_bgr"])
    st.download_button(
        label="💾 Download Highlighted Forensic Result (PNG)",
        data=download_bytes,
        file_name="forensic_inspection_result.png",
        mime="image/png",
        use_container_width=True
    )

    # Detected Regions Table
    if results["total_regions"] > 0:
        st.markdown("### 📍 Detected Regions Inventory")
        reg_data = []
        for reg in results["detected_regions"]:
            reg_data.append({
                "Region ID": f"#{reg['id']}",
                "X Offset (px)": reg["x"],
                "Y Offset (px)": reg["y"],
                "Width (px)": reg["w"],
                "Height (px)": reg["h"],
                "Contour Area (px²)": f"{reg['area']:.1f}"
            })
        st.dataframe(reg_data, use_container_width=True)
    else:
        st.success("✅ **No discrepancy detected above the selected area threshold.** The documents match under current filter settings.")

    # Forensic Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Forensic Disclaimer & Advisory:</strong><br>
        This document comparison system identifies pixel-level intensity differences between two scanned documents.
        Discrepancies highlight visual changes and <strong>do not constitute legal proof of forgery or fraudulent intent</strong>.
        Variations can occur due to scanner misalignments, paper folds, optical sensor noise, and lossy compression artifacts.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
