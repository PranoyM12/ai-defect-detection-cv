#!/usr/bin/env python3
"""
run_defect_detection.py
========================
Edit the paths below and run:  python run_defect_detection.py
"""

from production_defect_detection import (
    process_images_to_dataset, compare_models, train_final_model,
    predict_on_new_image, get_feature_names
)
import pandas as pd
import os

# ============================================================================
# CONFIGURATION — EDIT THESE PATHS
# ============================================================================

# Folder that contains Positive/ and Negative/ subfolders
TRAINING_FOLDER = 'training_images/'   # ← your structure is already correct

# Folder with images to run predictions on (no subfolders needed here)
TEST_FOLDER = 'test_images/'           # ← change if different

DATASET_CSV = 'my_defect_dataset.csv'

# manual_label is ignored when Positive/Negative subfolders are present
# (labels come from the folder names automatically)
MANUAL_LABELING = False

# ============================================================================
# STEP 1: BUILD DATASET
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: BUILDING DATASET FROM IMAGES")
print("=" * 70)

if os.path.exists(DATASET_CSV):
    print(f"\n✓ Found existing dataset: {DATASET_CSV}")
    response = input("Rebuild it? (y/n): ")
    if response.lower() != 'y':
        print("Using existing dataset.\n")
        df = pd.read_csv(DATASET_CSV)
    else:
        df = process_images_to_dataset(
            image_folder=TRAINING_FOLDER,
            output_csv=DATASET_CSV,
            manual_label=MANUAL_LABELING
        )
else:
    if not os.path.isdir(TRAINING_FOLDER):
        print(f"\n❌ ERROR: Training folder not found: {TRAINING_FOLDER}")
        print("Create the folder (with Positive/ and Negative/ subfolders) and try again.")
        exit(1)

    df = process_images_to_dataset(
        image_folder=TRAINING_FOLDER,
        output_csv=DATASET_CSV,
        manual_label=MANUAL_LABELING
    )

if df.empty:
    print("❌ Dataset is empty. Check your image folders.")
    exit(1)

print(f"✓ Dataset loaded: {len(df)} samples")
print(f"  Defects (1): {sum(df['label'] == 1)}")
print(f"  Normal  (0): {sum(df['label'] == 0)}")

if len(df) < 10:
    print("\n⚠ Very few samples (<10). Results may be unreliable.")
    print("  Recommendation: add more images (50+ per class recommended).")

if sum(df['label'] == 1) == 0 or sum(df['label'] == 0) == 0:
    print("\n❌ All samples have the same label — cannot train a binary classifier.")
    print("  Make sure BOTH Positive/ and Negative/ subfolders have images.")
    exit(1)

# ============================================================================
# STEP 2: COMPARE MODELS (optional)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: COMPARING MODELS")
print("=" * 70)

response = input("\nCompare models with cross-validation? (y/n): ")

if response.lower() == 'y':
    X = df[get_feature_names()].values
    y = df['label'].values
    results, best_model_name, scaler = compare_models(X, y, cv_folds=5)
    print(f"\n🏆 Best model: {best_model_name}")
else:
    best_model_name = 'Random Forest'
    print(f"\nSkipping comparison. Default model: {best_model_name}")

# ============================================================================
# STEP 3: TRAIN FINAL MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: TRAINING FINAL MODEL")
print("=" * 70)

model_filename = f"{best_model_name.lower().replace(' ', '_')}_model.pkl"

if os.path.exists(model_filename):
    print(f"\n✓ Found existing model: {model_filename}")
    response = input("Retrain it? (y/n): ")
    if response.lower() != 'y':
        print("Using existing model.\n")
    else:
        model, scaler = train_final_model(
            csv_path=DATASET_CSV,
            model_name=best_model_name,
            test_size=0.2
        )
else:
    model, scaler = train_final_model(
        csv_path=DATASET_CSV,
        model_name=best_model_name,
        test_size=0.2
    )

print(f"\n✓ Model ready: {model_filename}")

# ============================================================================
# STEP 4: PREDICT ON TEST IMAGES
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: MAKING PREDICTIONS ON TEST IMAGES")
print("=" * 70)

if not os.path.isdir(TEST_FOLDER):
    print(f"\n⚠ Test folder not found: {TEST_FOLDER}")
    print("Skipping predictions. Create the folder and add images to use this step.")
else:
    test_images = [f for f in os.listdir(TEST_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if len(test_images) == 0:
        print(f"\n⚠ No images found in {TEST_FOLDER}")
    else:
        print(f"\nFound {len(test_images)} test images.")
        response = input("Run predictions on all of them? (y/n): ")

        if response.lower() == 'y':
            results_summary = []

            for idx, img_name in enumerate(test_images):
                print(f"\nProcessing {idx + 1}/{len(test_images)}: {img_name}")
                image_path = os.path.join(TEST_FOLDER, img_name)
                output_path = f"prediction_{img_name}"

                defects, normal = predict_on_new_image(
                    image_path=image_path,
                    model_path=model_filename,
                    output_path=output_path
                )
                results_summary.append({
                    'image': img_name,
                    'defects': defects,
                    'normal': normal
                })

            print("\n" + "=" * 70)
            print("PREDICTION SUMMARY")
            print("=" * 70)
            print(f"{'Image':<35} {'Defects':<10} {'Normal':<10}")
            print("-" * 70)
            for r in results_summary:
                print(f"{r['image']:<35} {r['defects']:<10} {r['normal']:<10}")
            print("=" * 70)

# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 70)
print("ALL DONE! ✨")
print("=" * 70)
print(f"\n  ✓ {DATASET_CSV}     — feature dataset")
print(f"  ✓ {model_filename}  — trained model")
print("\nTo predict on a single image later:")
print("  from production_defect_detection import predict_on_new_image")
print(f"  predict_on_new_image('new.jpg', '{model_filename}')")
print("\n🎉 Happy defect detecting!\n")
