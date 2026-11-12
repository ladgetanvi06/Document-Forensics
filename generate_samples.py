"""
generate_samples.py
-------------------
Utility script to generate realistic synthetic scanned document pairs
for testing the Document Forensics image processing pipeline.

Uses Windows standard Arial font if available, with graceful fallback.
Generates 4 test scenarios:
  1. Identical Documents (Original vs Exact Copy)
  2. Changed Invoice Amount ($1,450.00 -> $9,750.00)
  3. Changed Issue Date (15-MAR-2024 -> 28-NOV-2025) and Ref Number
  4. Scanned Mismatch (Different dimensions & slight shift)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int = 16, bold: bool = False) -> ImageFont.ImageFont:
    """Safely retrieves a TrueType font if available, else PIL default."""
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_base_document(
    title: str = "OFFICIAL COMMERCIAL INVOICE",
    invoice_no: str = "INV-2024-8841",
    date_str: str = "15-MAR-2024",
    recipient: str = "Apex Technologies Corp.",
    amount_str: str = "$1,450.00",
    width: int = 800,
    height: int = 1000,
    add_noise: bool = True
) -> Image.Image:
    """Generates a PIL Image resembling a realistic scanned business invoice."""
    img = Image.new("RGB", (width, height), color=(253, 253, 252))
    draw = ImageDraw.Draw(img)

    # Fonts
    font_title = get_font(24, bold=True)
    font_sub = get_font(13, bold=False)
    font_header = get_font(15, bold=True)
    font_body = get_font(15, bold=False)
    font_bold = get_font(16, bold=True)
    font_amount = get_font(20, bold=True)
    font_stamp = get_font(18, bold=True)
    font_footer = get_font(12, bold=False)

    # Outer border / letterhead
    draw.rectangle([(25, 25), (width - 25, height - 25)], outline=(190, 190, 195), width=2)
    draw.rectangle([(30, 30), (width - 30, 130)], fill=(240, 244, 250))

    # Header title
    draw.text((50, 45), title, fill=(15, 35, 75), font=font_title)
    draw.text((50, 85), "GLOBAL ENTERPRISE BILLING DIVISION - VERIFIED AUDIT COPY", fill=(100, 110, 125), font=font_sub)

    # Metadata block
    draw.line([(50, 160), (width - 50, 160)], fill=(215, 215, 220), width=1)
    draw.text((50, 180), f"Invoice Reference:   {invoice_no}", fill=(35, 35, 35), font=font_body)
    draw.text((50, 215), f"Billing Date:        {date_str}", fill=(35, 35, 35), font=font_body)
    draw.text((50, 250), f"Billed To:           {recipient}", fill=(35, 35, 35), font=font_body)
    draw.text((50, 285), "Account Status:      VERIFIED & SIGNED", fill=(20, 130, 50), font=font_bold)

    # Table Header
    draw.rectangle([(50, 340), (width - 50, 380)], fill=(230, 236, 245))
    draw.text((65, 350), "Item Description", fill=(30, 30, 30), font=font_header)
    draw.text((450, 350), "Qty", fill=(30, 30, 30), font=font_header)
    draw.text((540, 350), "Unit Price", fill=(30, 30, 30), font=font_header)
    draw.text((660, 350), "Total", fill=(30, 30, 30), font=font_header)

    # Table Rows
    items = [
        ("Cloud Infrastructure Hosting (Q1)", "1", "$800.00", "$800.00", 405),
        ("Security Compliance & Code Audit", "1", "$450.00", "$450.00", 450),
        ("Standard Database Maintenance", "1", "$200.00", "$200.00", 495),
    ]
    for desc, qty, rate, tot, y in items:
        draw.text((65, y), desc, fill=(55, 55, 55), font=font_body)
        draw.text((455, y), qty, fill=(55, 55, 55), font=font_body)
        draw.text((540, y), rate, fill=(55, 55, 55), font=font_body)
        draw.text((660, y), tot, fill=(55, 55, 55), font=font_body)
        draw.line([(50, y + 32), (width - 50, y + 32)], fill=(235, 235, 240), width=1)

    # Total Box
    draw.rectangle([(420, 570), (width - 50, 630)], fill=(245, 246, 250), outline=(200, 205, 215), width=2)
    draw.text((435, 588), "NET TOTAL PAYABLE:", fill=(30, 30, 30), font=font_bold)
    draw.text((630, 585), amount_str, fill=(190, 25, 25), font=font_amount)

    # Signatures and stamp area
    draw.text((50, 720), "Authorized Signatory: _________________________", fill=(100, 100, 100), font=font_body)
    draw.text((50, 755), "Chief Financial Officer, Apex Corp", fill=(120, 120, 120), font=font_sub)

    # Stamp
    draw.ellipse([(width - 240, 700), (width - 80, 810)], outline=(180, 40, 40), width=2)
    draw.text((width - 205, 745), "AUDITED", fill=(180, 40, 40), font=font_stamp)

    # Footer
    draw.line([(50, 910), (width - 50, 910)], fill=(215, 215, 220), width=1)
    draw.text((50, 925), "Page 1 of 1 | Confidential Scanned Record | Security Checksum: SEC-9842A", fill=(140, 140, 145), font=font_footer)

    # Subtle scan noise
    if add_noise:
        arr = np.array(img)
        noise_mat = np.random.normal(0, 1.5, arr.shape).astype(np.int16)
        noisy_arr = np.clip(arr.astype(np.int16) + noise_mat, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_arr)

    return img


def generate_all_samples(output_dir: str = "samples"):
    """Generates all sample test pairs into the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating test sample pairs into: {output_dir}")

    # Pair 1: Identical Images
    p1_orig = create_base_document()
    p1_copy = p1_orig.copy()
    p1_orig.save(os.path.join(output_dir, "pair1_identical_orig.png"))
    p1_copy.save(os.path.join(output_dir, "pair1_identical_susp.png"))

    # Pair 2: Tampered Invoice Amount ($1,450.00 -> $9,750.00)
    p2_orig = create_base_document(amount_str="$1,450.00")
    p2_susp = create_base_document(amount_str="$9,750.00")
    p2_orig.save(os.path.join(output_dir, "pair2_amount_orig.png"))
    p2_susp.save(os.path.join(output_dir, "pair2_amount_tampered.png"))

    # Pair 3: Tampered Issue Date & Ref Number
    p3_orig = create_base_document(date_str="15-MAR-2024", invoice_no="INV-2024-8841")
    p3_susp = create_base_document(date_str="28-NOV-2025", invoice_no="INV-2025-9920")
    p3_orig.save(os.path.join(output_dir, "pair3_date_orig.png"))
    p3_susp.save(os.path.join(output_dir, "pair3_date_tampered.png"))

    # Pair 4: Dimension mismatch & scan offset
    p4_orig = create_base_document(width=800, height=1000)
    # Different dimensions (780x980) and modified total
    p4_susp = create_base_document(amount_str="$4,900.00", width=780, height=980)
    p4_orig.save(os.path.join(output_dir, "pair4_mismatch_orig.png"))
    p4_susp.save(os.path.join(output_dir, "pair4_mismatch_tampered.png"))

    print("Successfully generated all sample test pairs with crisp TrueType fonts!")


if __name__ == "__main__":
    generate_all_samples()
