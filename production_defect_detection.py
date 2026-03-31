"""
Production-Grade Defect Detection System
==========================================
Features:
1. Multi-image dataset processing (supports Positive/Negative subfolders)
2. CSV export/import for datasets
3. Manual labeling interface
4. Advanced feature extraction (Hu moments, GLCM, skeleton)
5. Multiple model comparison
6. Cross-validation
7. Proper train/test separation
"""

import cv2
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: ADVANCED FEATURE EXTRACTION
# ============================================================================

def extract_advanced_features(contour, image):
    """
    Extract comprehensive feature set including:
    - Shape features (area, perimeter, etc.)
    - Hu moments (rotation-invariant shape descriptors)
    - GLCM texture features (contrast, homogeneity, energy, correlation)
    - Skeleton features (crack thinness measure)
    - Intensity statistics
    """
    x, y_coord, w, h = cv2.boundingRect(contour)

    # Bounds check
    if (y_coord + h > image.shape[0] or x + w > image.shape[1] or
            w == 0 or h == 0 or y_coord < 0 or x < 0):
        return None

    roi = image[y_coord:y_coord + h, x:x + w]

    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return None

    # === BASIC SHAPE FEATURES ===
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return None

    aspect_ratio = w / h
    rect_area = w * h
    extent = area / rect_area if rect_area != 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    # Circularity (4π*area/perimeter²)
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Compactness (perimeter²/area)
    compactness = (perimeter * perimeter) / area if area > 0 else 0

    # === HU MOMENTS (rotation-invariant shape) ===
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # === TEXTURE FEATURES (GLCM) ===
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if gray_roi.shape[0] < 5 or gray_roi.shape[1] < 5:
        gray_roi = cv2.resize(gray_roi, (20, 20))

    glcm = graycomatrix(
        gray_roi,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    # === SKELETON FEATURES (for thin cracks) ===
    _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)
    skeleton = skeletonize(binary)
    skeleton_length = np.sum(skeleton)
    skeleton_density = skeleton_length / area if area > 0 else 0

    # === EDGE FEATURES ===
    edges_roi = cv2.Canny(gray_roi, 30, 100)
    edge_density = np.sum(edges_roi) / (w * h * 255) if (w * h) > 0 else 0

    # === INTENSITY FEATURES ===
    mean_intensity = np.mean(gray_roi)
    std_intensity = np.std(gray_roi)
    min_intensity = np.min(gray_roi)
    max_intensity = np.max(gray_roi)
    variance = np.var(gray_roi)
    intensity_range = max_intensity - min_intensity

    # === COMPILE FEATURE VECTOR ===
    features = [
        # Shape (9 features)
        area, perimeter, aspect_ratio, extent, solidity,
        circularity, compactness, w, h,

        # Hu moments (7 features)
        hu_moments[0], hu_moments[1], hu_moments[2], hu_moments[3],
        hu_moments[4], hu_moments[5], hu_moments[6],

        # GLCM texture (5 features)
        contrast, dissimilarity, homogeneity, energy, correlation,

        # Skeleton (2 features)
        skeleton_length, skeleton_density,

        # Edge (1 feature)
        edge_density,

        # Intensity (6 features)
        mean_intensity, std_intensity, min_intensity, max_intensity,
        variance, intensity_range
    ]

    return np.array(features)


def get_feature_names():
    """Return feature names for the dataset CSV"""
    return [
        # Shape
        'area', 'perimeter', 'aspect_ratio', 'extent', 'solidity',
        'circularity', 'compactness', 'width', 'height',
        # Hu moments
        'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7',
        # GLCM
        'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
        'glcm_energy', 'glcm_correlation',
        # Skeleton
        'skeleton_length', 'skeleton_density',
        # Edge
        'edge_density',
        # Intensity
        'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
        'variance', 'intensity_range'
    ]


# ============================================================================
# STEP 2: MULTI-IMAGE DATASET PROCESSING
# Supports both flat folder and Positive/Negative subfolder structure
# ============================================================================

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def _collect_images_from_folder(image_folder):
    """
    Collect images with their ground-truth labels.

    Two supported structures:
      1. Flat folder  →  all images go through rule-based labeling
      2. Subfolder structure:
            image_folder/
            ├── Positive/   (label = 1, i.e. DEFECT)
            └── Negative/   (label = 0, i.e. NORMAL)

    Returns: list of (image_path, ground_truth_label_or_None)
             ground_truth_label is an int when known from folder name, else None.
    """
    pos_dir = os.path.join(image_folder, 'Positive')
    neg_dir = os.path.join(image_folder, 'Negative')

    # --- Subfolder mode ---
    if os.path.isdir(pos_dir) or os.path.isdir(neg_dir):
        print("  Detected Positive/Negative subfolder structure.")
        entries = []
        for subdir, folder_label in [(pos_dir, 1), (neg_dir, 0)]:
            if not os.path.isdir(subdir):
                print(f"  ⚠ Subfolder not found, skipping: {subdir}")
                continue
            files = [f for f in os.listdir(subdir)
                     if f.lower().endswith(IMAGE_EXTENSIONS)]
            label_name = "Defect (Positive)" if folder_label == 1 else "Normal (Negative)"
            print(f"  {label_name}: {len(files)} images")
            for f in files:
                entries.append((os.path.join(subdir, f), folder_label))
        return entries

    # --- Flat folder mode ---
    files = [f for f in os.listdir(image_folder)
             if f.lower().endswith(IMAGE_EXTENSIONS)]
    print(f"  Flat folder mode: {len(files)} images (rule-based labeling will be used).")
    return [(os.path.join(image_folder, f), None) for f in files]


def _preprocess_image(image):
    """Shared preprocessing pipeline used for both training and inference."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges1 = cv2.Canny(blur, 20, 60)
    edges2 = cv2.Canny(blur, 30, 100)
    edges_combined = cv2.bitwise_or(edges1, edges2)
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges_combined, kernel_small, iterations=1)
    edges_final = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE,
                                   kernel_medium, iterations=2)
    return edges_final


def process_images_to_dataset(image_folder, output_csv='dataset.csv', manual_label=False):
    """
    Process multiple images and create a dataset CSV.

    Args:
        image_folder:  Path to folder containing images.
                       Can be a flat folder OR contain Positive/ and Negative/ subfolders.
                       When subfolders are present the ground-truth label is taken from
                       the folder name (Positive → 1, Negative → 0) and manual_label is
                       ignored unless you explicitly want to override per-contour labels.
        output_csv:    Output CSV filename.
        manual_label:  Only used in flat-folder mode. If True, prompt for labels.
    """
    print(f"\n{'=' * 60}")
    print("PROCESSING IMAGES TO DATASET")
    print(f"{'=' * 60}\n")

    all_features = []
    all_labels = []
    all_filenames = []

    entries = _collect_images_from_folder(image_folder)

    if len(entries) == 0:
        print("❌ No images found. Check your folder path.")
        return pd.DataFrame()

    print(f"\nTotal images to process: {len(entries)}\n")

    for idx, (image_path, folder_label) in enumerate(entries):
        filename = os.path.basename(image_path)
        print(f"Processing {idx + 1}/{len(entries)}: {filename}", end='')
        if folder_label is not None:
            print(f"  [ground-truth: {'DEFECT' if folder_label == 1 else 'NORMAL'}]")
        else:
            print()

        image = cv2.imread(image_path)
        if image is None:
            print(f"  ⚠ Could not load image, skipping...")
            continue

        edges_final = _preprocess_image(image)
        contours, _ = cv2.findContours(edges_final, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        print(f"  Found {len(contours)} contours")

        image_defects = 0
        image_normal = 0

        for cnt_idx, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 20:
                features = extract_advanced_features(cnt, image)

                if features is not None:
                    # ---- Label assignment ----
                    if folder_label is not None:
                        # Ground-truth from subfolder name – use it for ALL contours
                        label = folder_label

                    elif manual_label:
                        # Interactive per-contour labeling (flat-folder mode only)
                        image_display = image.copy()
                        cv2.drawContours(image_display, [cnt], -1, (0, 255, 0), 2)
                        cv2.imshow('Contour - Is this a defect?', image_display)
                        cv2.waitKey(1)
                        while True:
                            label_input = input(
                                f"  Contour {cnt_idx + 1}: Is this a DEFECT? (1=yes, 0=no): ")
                            if label_input in ['0', '1']:
                                label = int(label_input)
                                break
                            print("    Invalid input. Enter 0 or 1.")
                        cv2.destroyAllWindows()

                    else:
                        # Rule-based labeling (flat-folder mode only)
                        area = cv2.contourArea(cnt)
                        edge_density = features[23]   # edge_density feature index
                        circularity = features[5]     # circularity feature index
                        variance = features[28]       # variance feature index

                        if ((edge_density > 0.04 and area > 50) or
                                (circularity < 0.3 and area > 80) or
                                (variance > 500 and area > 60)):
                            label = 1  # Defect
                        else:
                            label = 0  # Normal

                    all_features.append(features)
                    all_labels.append(label)
                    all_filenames.append(filename)

                    if label == 1:
                        image_defects += 1
                    else:
                        image_normal += 1

        print(f"  → Contours labelled: {image_defects} defect, {image_normal} normal\n")

    if len(all_features) == 0:
        print("❌ No features extracted. Check your images.")
        return pd.DataFrame()

    # Create DataFrame
    feature_names = get_feature_names()
    df = pd.DataFrame(all_features, columns=feature_names)
    df['label'] = all_labels
    df['filename'] = all_filenames

    df.to_csv(output_csv, index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ Dataset saved to: {output_csv}")
    print(f"   Total samples : {len(df)}")
    print(f"   Defects (1)   : {sum(df['label'] == 1)}")
    print(f"   Normal  (0)   : {sum(df['label'] == 0)}")
    print(f"{'=' * 60}\n")

    return df


# ============================================================================
# STEP 3: MODEL COMPARISON & CROSS-VALIDATION
# ============================================================================

def compare_models(X, y, cv_folds=5):
    """
    Compare multiple models using cross-validation.
    Returns best model name and comparison results.
    """
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON WITH CROSS-VALIDATION")
    print(f"{'=' * 60}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        print(f"  {cv_folds}-Fold CV Accuracy: {cv_scores.mean():.4f} "
              f"(+/- {cv_scores.std():.4f})")
        print(f"  Individual folds: {[f'{s:.4f}' for s in cv_scores]}\n")

    best_model_name = max(results, key=lambda k: results[k]['cv_mean'])
    print(f"{'=' * 60}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   CV Accuracy : {results[best_model_name]['cv_mean']:.4f}")
    print(f"{'=' * 60}\n")

    return results, best_model_name, scaler


# ============================================================================
# STEP 4: FINAL TRAINING & EVALUATION
# ============================================================================

def train_final_model(csv_path, model_name='Random Forest', test_size=0.2):
    """
    Load dataset from CSV and train the final model with a proper train/test split.
    Saves the model + scaler as a .pkl file.
    """
    print(f"\n{'=' * 60}")
    print("TRAINING FINAL MODEL")
    print(f"{'=' * 60}\n")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} samples")
    print(f"  Defects (1): {sum(df['label'] == 1)}")
    print(f"  Normal  (0): {sum(df['label'] == 0)}\n")

    feature_cols = get_feature_names()
    X = df[feature_cols].values
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Train set : {len(X_train)} samples")
    print(f"Test set  : {len(X_test)} samples\n")

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    if model_name not in models:
        print(f"⚠ Unknown model '{model_name}'. Defaulting to Random Forest.")
        model_name = 'Random Forest'

    model = models[model_name]

    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n{'=' * 60}")
    print("TEST SET EVALUATION")
    print(f"{'=' * 60}\n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Defect']))

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print(f"\n{'=' * 60}")
        print("TOP 10 MOST IMPORTANT FEATURES")
        print(f"{'=' * 60}")
        for i, idx in enumerate(indices):
            print(f"{i + 1}. {feature_cols[idx]}: {importances[idx]:.4f}")

    model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

    print(f"\n✅ Model saved to: {model_filename}\n")
    return model, scaler


# ============================================================================
# STEP 5: PREDICT ON NEW IMAGE
# ============================================================================

def predict_on_new_image(image_path, model_path, output_path='prediction_result.png'):
    """
    Load trained model and predict defects on a new image.
    Returns (defect_count, normal_count).
    """
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load {image_path}")
        return 0, 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Same preprocessing as training
    edges_final = _preprocess_image(image)
    contours, _ = cv2.findContours(edges_final, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    output = image_rgb.copy()
    defect_count = 0
    normal_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            features = extract_advanced_features(cnt, image)
            if features is not None:
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                if prediction == 1:
                    cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
                    defect_count += 1
                else:
                    cv2.drawContours(output, [cnt], -1, (0, 255, 0), 1)
                    normal_count += 1

    print(f"\n{'=' * 60}")
    print("PREDICTION RESULTS")
    print(f"{'=' * 60}")
    print(f"Image           : {image_path}")
    print(f"Defects (RED)   : {defect_count}")
    print(f"Normal  (GREEN) : {normal_count}")
    print(f"{'=' * 60}\n")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title(f"Predictions — {defect_count} defects (RED)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Result saved to: {output_path}\n")

    return defect_count, normal_count
