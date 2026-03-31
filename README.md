# AI-Based Defect Detection System (Computer Vision + Machine Learning)

## 🔍 Overview

This project implements a hybrid **Computer Vision + Machine Learning pipeline** to detect cracks and surface defects in images.

It combines classical image processing with feature-engineered ML models to achieve robust detection across different surface types.

---

## ⚙️ Approach

### 1. Computer Vision Pipeline

* Grayscale conversion
* Gaussian blur (noise reduction)
* Canny edge detection
* Morphological operations
* Contour extraction

### 2. Feature Engineering

For each detected contour:

* Shape features (area, perimeter, aspect ratio, solidity)
* Texture features (GLCM, variance)
* Edge-based features (edge density)
* Structural features (skeleton-based crack analysis)

### 3. Machine Learning

* Model: Random Forest Classifier
* Trained on large-scale dataset (~75,000 images)
* Handles classification of defect vs non-defect regions

### 4. False Positive Reduction

* Shape filtering (elongated structures only)
* Edge strength filtering
* Confidence-based prediction threshold

---

## 📊 Results

The model successfully:

* Detects real cracks and defects
* Reduces false positives from textured surfaces
* Generalizes across different materials (walls, concrete, etc.)

### Sample Outputs:

(Add 2–3 images from sample_outputs folder here)

---

## 🚀 How to Run

```bash
python run_defect_detection.py
```

---

## 📁 Project Structure

```
defect-detection-cv/
├── production_defect_detection.py
├── run_defect_detection.py
├── random_forest_model.pkl
├── test_images/
├── sample_outputs/
├── my_defect_dataset.csv
└── README.md
```

---

## 🧠 Key Learning

* Designed hybrid CV + ML system
* Handled real-world issue of false positives
* Used feature engineering to improve model performance
* Built scalable dataset pipeline

---

## 🔮 Future Improvements

* Deep learning (CNN / U-Net)
* Real-time defect detection
* Deployment as API or web app


## ⚠️ Note

The trained model file (`.pkl`) is not included due to size limitations.

To use the project:
1. Generate dataset using the provided pipeline
2. Train the model using the training script
