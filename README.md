# Document Forensics: Suspicious Edit Detection in Scanned Documents

> **SIP (Self-Instruction / Semester Internship Project)**  
> **Domain:** Digital Image Processing & Document Forensics  
> **Technologies:** Python, OpenCV (`opencv-python`), Streamlit, NumPy, Pillow  

---

## 📌 Project Overview

This project is an **educational image-processing prototype** designed to detect and highlight suspicious localized edits between two scanned document images:
1. **Reference (Authentic) Document**: The known authentic original.
2. **Suspected (Questioned) Document**: The document suspected of unauthorized modification (e.g., altered invoice amounts, forged dates, or revised clauses).

The application compares the two documents using classical computer vision techniques—**without deep learning, model training, databases, or cloud APIs**—providing full transparency into every step of the image processing pipeline.

---

## 📁 Project Folder Structure

```text
sip/
│
├── app.py                   # Streamlit web application dashboard
├── processing.py            # Core OpenCV image processing pipeline module
├── generate_samples.py      # Synthetic test documents generator (invoices & certificates)
├── test_pipeline.py         # Automated test suite for validation across all test pairs
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation and setup guide
│
└── samples/                 # Sample document pairs for testing & demonstration
    ├── pair1_identical_orig.png
    ├── pair1_identical_susp.png
    ├── pair2_amount_orig.png
    ├── pair2_amount_tampered.png
    ├── pair3_date_orig.png
    ├── pair3_date_tampered.png
    ├── pair4_mismatch_orig.png
    └── pair4_mismatch_tampered.png
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Install Required Dependencies
Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

*(Installed packages: `opencv-python`, `streamlit`, `numpy`, `pillow`)*

---

## 🚀 How to Run the Application

### 1. (Optional) Generate Fresh Test Samples
```bash
python generate_samples.py
```
This generates 4 realistic sample document pairs in the `samples/` directory.

### 2. (Optional) Run the Pipeline Test Suite
```bash
python test_pipeline.py
```
This verifies that:
- Identical documents produce **0 false positive regions**.
- Altered invoice amounts are accurately detected.
- Altered dates and reference numbers are localized.
- Dimension mismatches trigger automatic alignment and user warnings.

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Once launched, open your browser and navigate to:
**`http://localhost:8501`**

---

## 🖥️ Using the Web Dashboard

1. **Document Upload**: Upload the authentic **Original Document** and the questioned **Suspected Document** using the sidebar uploaders (PNG, JPG, TIFF, BMP).
2. **Adjust Parameters (Optional)**:
   - **Filter Type**: Choose between *Gaussian* (smooth blurring) and *Median* (ideal for scanner salt-and-pepper noise).
   - **Kernel Size**: Slider for filter neighborhood (default $3\times3$).
   - **Threshold**: Adjust the minimum intensity change required to register as a difference, or toggle *Otsu's Automatic Thresholding*.
   - **Dilation Iterations**: Controls how adjacent character strokes are merged into solid word boxes.
   - **Minimum Region Area**: Suppresses tiny scanner dust particles below the specified pixel area.
3. **Compare Documents**: Click the **Run Document Comparison** button.
4. **Inspect Results**: Review the side-by-side inspection cards and the detected regions inventory table.
5. **Download Results**: Click **Download Highlighted Forensic Result (PNG)** to save the annotated document.

---

## 🔬 Image Processing Pipeline Explained

| Stage | Operation | OpenCV Function | Purpose |
|---|---|---|---|
| **1. Dimension Alignment** | Resizing / Matching | `cv2.resize` | Ensures both matrices have matching $(H, W)$ dimensions. Discloses an alignment warning to the user. |
| **2. Grayscale Conversion** | Color Reduction | `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` | Reduces 3 color channels to 1 intensity channel ($Y = 0.299R + 0.587G + 0.114B$). |
| **3. Spatial Filtering** | Noise Smoothing | `cv2.GaussianBlur` or `cv2.medianBlur` | Eliminates scanner sensor noise, dust speckles, and paper grain. |
| **4. Absolute Difference** | Subtraction | `cv2.absdiff(orig, susp)` | Calculates $\|I_1(x, y) - I_2(x, y)\|$; identical areas become 0 (black). |
| **5. Thresholding** | Binarization | `cv2.threshold` | Converts difference map to binary mask ($0$ or $255$) using manual or Otsu's method. |
| **6. Morphology** | Dilation & Closing | `cv2.dilate` & `cv2.morphologyEx` | Merges fragmented character strokes into cohesive word blocks; removes isolated pixels. |
| **7. Contour Detection** | Boundary Tracing | `cv2.findContours` & `cv2.boundingRect` | Identifies connected components $\ge \text{min\_area}$ and draws bounding boxes with badge numbers. |

---

## ⚠️ Limitations & False Positives / Negatives

- **Scanner Misalignment (False Positives)**: If a document is scanned at a slight rotational angle (skew) or shifted by even 2-3 pixels, letter margins may light up as differences.
- **Paper Folds & Creases (False Positives)**: Physical creases on the paper surface create shadows that produce intensity differences.
- **Lossy JPEG Compression**: Artifacts around text edges can register as minor differences.
- **Ink / Font Matching (False Negatives)**: If an edit is executed using identical font dimensions and exact gray level, very subtle alterations may fall below the difference threshold.
- **Academic Disclaimer**: This system identifies pixel-level discrepancies for educational investigation; it does not certify legal document authenticity.
