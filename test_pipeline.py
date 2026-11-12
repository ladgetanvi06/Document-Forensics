"""
test_pipeline.py
----------------
Quick smoke-test script to test the OpenCV pipeline against all generated test pairs.
"""

from processing import run_document_forensics_pipeline, load_image_from_path


def main():
    print("=" * 60)
    print("RUNNING DOCUMENT FORENSICS PIPELINE TEST SUITE")
    print("=" * 60)

    # Test 1: Identical documents
    orig1 = load_image_from_path("samples/pair1_identical_orig.png")
    susp1 = load_image_from_path("samples/pair1_identical_susp.png")
    res1 = run_document_forensics_pipeline(orig1, susp1)
    print(f"\n[Test 1: Identical Documents]")
    print(f"  Detected Regions: {res1['total_regions']}")
    print(f"  Is Resized: {res1['is_resized']}")
    assert res1['total_regions'] == 0, f"Expected 0 changes, got {res1['total_regions']}"
    print("  => PASSED (Zero false positives on identical docs)")

    # Test 2: Altered Invoice Amount
    orig2 = load_image_from_path("samples/pair2_amount_orig.png")
    susp2 = load_image_from_path("samples/pair2_amount_tampered.png")
    res2 = run_document_forensics_pipeline(orig2, susp2)
    print(f"\n[Test 2: Altered Invoice Amount]")
    print(f"  Detected Regions: {res2['total_regions']}")
    print(f"  Bounding Boxes: {res2['detected_regions']}")
    assert res2['total_regions'] >= 1, "Expected at least 1 altered region"
    print("  => PASSED (Tampered amount accurately detected)")

    # Test 3: Altered Date and Reference
    orig3 = load_image_from_path("samples/pair3_date_orig.png")
    susp3 = load_image_from_path("samples/pair3_date_tampered.png")
    res3 = run_document_forensics_pipeline(orig3, susp3)
    print(f"\n[Test 3: Altered Date & Ref Number]")
    print(f"  Detected Regions: {res3['total_regions']}")
    print(f"  Bounding Boxes: {res3['detected_regions']}")
    assert res3['total_regions'] >= 1, "Expected at least 1 altered region"
    print("  => PASSED (Tampered date/ref accurately detected)")

    # Test 4: Dimension Mismatch
    orig4 = load_image_from_path("samples/pair4_mismatch_orig.png")
    susp4 = load_image_from_path("samples/pair4_mismatch_tampered.png")
    res4 = run_document_forensics_pipeline(orig4, susp4)
    print(f"\n[Test 4: Dimension Mismatch & Content Shift]")
    print(f"  Detected Regions: {res4['total_regions']}")
    print(f"  Is Resized: {res4['is_resized']}")
    print(f"  Resizing Warning Disclosed: {res4['resized_warning'] is not None}")
    assert res4['is_resized'] is True, "Expected resizing flag to be True"
    print("  => PASSED (Dimension mismatch handled and warned)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
