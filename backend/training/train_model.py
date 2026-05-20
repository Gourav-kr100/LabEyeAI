"""
Training script for Smart Lab Chair Monitoring System.
Learns reference patterns from labeled lab photos (correct vs misplaced).

Approach:
1. Use YOLOv8 to detect chairs in each labeled image
2. Extract spatial and visual features from chair arrangements
3. Augment training data with flips, rotations, and brightness variations
4. Train an SVM classifier with leave-one-group-out cross-validation
5. Save trained profile + SVM model for use in the analyzer
"""

import os
import sys
import json
import cv2
import numpy as np
import shutil
import pickle
from datetime import datetime
from collections import Counter

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

# ===== CONFIGURATION =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CORRECT_DIR = os.path.join(DATA_DIR, "correct")
MISPLACED_DIR = os.path.join(DATA_DIR, "misplaced")
PROFILE_PATH = os.path.join(PROJECT_ROOT, "models", "trained_profile.json")
SVM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "scene_classifier.pkl")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

CHAIR_CLASS_ID = 56

# Source images from user's lab photos (in Labproject root)
LAB_PHOTOS_DIR = os.path.dirname(PROJECT_ROOT)


def organize_training_data():
    """
    Organize user's lab photos into correct/misplaced directories.
    Based on user's labeling of their uploaded photos.
    """
    os.makedirs(CORRECT_DIR, exist_ok=True)
    os.makedirs(MISPLACED_DIR, exist_ok=True)

    # User-provided labels based on their feedback
    labeled_images = {
        # CORRECT: chairs properly tucked into desks
        "correct": [
            "Unknown.jpeg",
            "WhatsApp Image 2026-05-15 at 1.14.52 PM.jpeg",
            "WhatsApp Image 2026-05-15 at 1.15.00 PM.jpeg",
            "WhatsApp Image 2026-05-15 at 1.15.07 PM.jpeg",
        ],
        # MISPLACED: chairs pulled out, scattered, not tucked in
        "misplaced": [
            "Unknown-1.jpeg",
            "Unknown-2.jpeg",
            "WhatsApp Image 2026-05-15 at 1.14.53 PM.jpeg",
            "WhatsApp Image 2026-05-15 at 1.14.53 PM-2.jpeg",
            "WhatsApp Image 2026-05-15 at 1.15.03 PM.jpeg",
            "Rotated_Chair_Sample.jpg"
        ],
    }

    # Do not wipe existing directories. Just organize new ones if needed.
    # The user can drop images directly into these folders to train.
    copied = {"correct": 0, "misplaced": 0}
    for label, filenames in labeled_images.items():
        dest_dir = CORRECT_DIR if label == "correct" else MISPLACED_DIR
        for fname in filenames:
            src = os.path.join(LAB_PHOTOS_DIR, fname)
            dst = os.path.join(dest_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied[label] += 1
                print(f"  ✅ [{label.upper()}] {fname}")
            elif not os.path.exists(src) and not os.path.exists(dst):
                print(f"  ⚠️  Not found: {fname}")

    print(f"\n📁 Organized: {copied['correct']} correct, {copied['misplaced']} misplaced")
    return copied


def augment_image(image):
    """Generate augmented versions of an image for training."""
    augmented = [image.copy()]  # 0: original

    # 1: Horizontal flip
    augmented.append(cv2.flip(image, 1))

    # 2-3: Brightness variations
    for alpha in [0.85, 1.15]:
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented.append(adjusted)

    # 4-5: Slight rotations
    h, w = image.shape[:2]
    for angle in [-3, 3]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # 6: Gaussian blur (simulates slight defocus)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmented.append(blurred)

    return augmented  # 7 versions total


def detect_chairs(image, model):
    """Detect chairs in image using YOLO (synchronized with analyzer.py)."""
    h, w = image.shape[:2]
    results = model(image, conf=0.15, classes=[CHAIR_CLASS_ID], verbose=False)
    chairs_raw = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bw, bh = x2 - x1, y2 - y1

            chairs_raw.append({
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "width": bw, "height": bh,
                "area": bw * bh,
                "confidence": conf,
                "aspect_ratio": bh / max(bw, 1)
            })

    # Filter overlapping chair detections (same 40px threshold as analyzer.py)
    chairs = []
    for c in chairs_raw:
        overlap = False
        for existing in chairs:
            if np.hypot(c["center"][0] - existing["center"][0], c["center"][1] - existing["center"][1]) < 40:
                overlap = True
                break
        if not overlap:
            chairs.append(c)

    return chairs


def extract_features(image, model):
    """
    Extract scene-level features from a lab image for training.
    Returns a feature dictionary describing the chair arrangement state.
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    img_area = h * w
    img_diagonal = np.sqrt(w**2 + h**2)

    chairs = detect_chairs(image, model)
    num_chairs = len(chairs)

    if num_chairs == 0:
        return None

    # === Chair spatial features ===
    chair_y = [c["center"][1] / h for c in chairs]
    chair_x = [c["center"][0] / w for c in chairs]
    chair_aspects = [c["aspect_ratio"] for c in chairs]

    # Inter-chair spacing (sorted by Y position)
    spacings = []
    if num_chairs >= 2:
        sorted_by_y = sorted(chairs, key=lambda c: c["center"][1])
        for i in range(1, len(sorted_by_y)):
            dist = np.sqrt(
                (sorted_by_y[i]["center"][0] - sorted_by_y[i-1]["center"][0])**2 +
                (sorted_by_y[i]["center"][1] - sorted_by_y[i-1]["center"][1])**2
            ) / img_diagonal
            spacings.append(dist)

    # === Column analysis ===
    max_column_x_dev = 0.0
    aisle_chair_ratio = 0.0
    chair_coverage = sum(c["area"] for c in chairs) / img_area

    if num_chairs >= 2:
        min_cx = min(c["center"][0] for c in chairs)
        max_cx = max(c["center"][0] for c in chairs)
        mid_cx = (min_cx + max_cx) / 2.0
        col_sep = max(max_cx - min_cx, 1.0)

        right = [c for c in chairs if c["center"][0] > mid_cx]
        left = [c for c in chairs if c["center"][0] <= mid_cx]

        x_devs = []
        if len(right) >= 2:
            r_median = float(np.median([c["center"][0] for c in right]))
            for c in right:
                dev = max(0, c["center"][0] - r_median) / col_sep
                x_devs.append(dev)
        if len(left) >= 2:
            l_median = float(np.median([c["center"][0] for c in left]))
            for c in left:
                dev = max(0, c["center"][0] - l_median) / col_sep
                x_devs.append(dev)

        if x_devs:
            max_column_x_dev = float(max(x_devs))
            aisle_chair_ratio = float(sum(1 for d in x_devs if d > 0.08) / num_chairs)

    # === Visual features ===
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Floor exposure (light floor in lower half)
    floor_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))
    lower_half = floor_mask[h // 2:, :]
    floor_exposure = float(np.sum(lower_half > 0) / max(lower_half.size, 1))

    # Edge density in lower half
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges[h // 2:, :] > 0) / max(edges[h // 2:, :].size, 1))

    # Blue desk divider exposure
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 80]), np.array([130, 255, 255]))
    blue_exposure = float(np.sum(blue_mask > 0) / max(blue_mask.size, 1))

    # === Build feature vector (keys sorted alphabetically for consistency) ===
    features = {
        "aisle_chair_ratio": aisle_chair_ratio,
        "aspect_variance": float(np.var(chair_aspects)) if len(chair_aspects) > 1 else 0.0,
        "avg_aspect_ratio": float(np.mean(chair_aspects)),
        "avg_spacing": float(np.mean(spacings)) if spacings else 0.0,
        "blue_desk_exposure": blue_exposure,
        "chair_coverage_ratio": float(chair_coverage),
        "edge_density": edge_density,
        "floor_exposure": floor_exposure,
        "max_column_x_deviation": max_column_x_dev,
        "std_chair_x": float(np.std(chair_x)) if len(chair_x) > 1 else 0.0,
        "std_chair_y": float(np.std(chair_y)) if len(chair_y) > 1 else 0.0,
        "std_spacing": float(np.std(spacings)) if len(spacings) > 1 else 0.0,
        "x_variance": float(np.var(chair_x)) if len(chair_x) > 1 else 0.0,
        "y_variance": float(np.var(chair_y)) if len(chair_y) > 1 else 0.0,
    }

    return features


def train():
    """Main training pipeline."""
    print("=" * 60)
    print("🧠 SMART LAB CHAIR MONITORING — TRAINING PIPELINE v2.0")
    print("=" * 60)

    # Step 1: Organize data
    print("\n📂 Step 1: Organizing training data...")
    organize_training_data()

    correct_files = sorted([f for f in os.listdir(CORRECT_DIR)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    misplaced_files = sorted([f for f in os.listdir(MISPLACED_DIR)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not correct_files or not misplaced_files:
        print("❌ Need at least 1 correct and 1 misplaced image!")
        return

    print(f"  Found {len(correct_files)} correct, {len(misplaced_files)} misplaced images")

    # Step 2: Load YOLO model
    print("\n🤖 Step 2: Loading YOLOv8 model...")
    model = YOLO(YOLO_MODEL_PATH)
    print("  ✅ Model loaded")

    # Step 3: Extract features with augmentation
    print("\n🔍 Step 3: Extracting features with data augmentation (7x per image)...")

    all_features = []
    all_labels = []
    all_groups = []  # group index for cross-validation (one group per original image)
    original_features = []
    original_labels = []

    group_idx = 0

    for fname in correct_files:
        path = os.path.join(CORRECT_DIR, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"  ⚠️ Could not read {fname}")
            continue

        augmented = augment_image(image)
        aug_count = 0
        for j, aug_img in enumerate(augmented):
            feats = extract_features(aug_img, model)
            if feats:
                all_features.append(feats)
                all_labels.append(0)  # 0 = correct
                all_groups.append(group_idx)
                aug_count += 1
                if j == 0:  # original image
                    original_features.append(feats)
                    original_labels.append(0)

        print(f"  [CORRECT] {fname}: {aug_count} augmented samples")
        group_idx += 1

    for fname in misplaced_files:
        path = os.path.join(MISPLACED_DIR, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"  ⚠️ Could not read {fname}")
            continue

        augmented = augment_image(image)
        aug_count = 0
        for j, aug_img in enumerate(augmented):
            feats = extract_features(aug_img, model)
            if feats:
                all_features.append(feats)
                all_labels.append(1)  # 1 = misplaced
                all_groups.append(group_idx)
                aug_count += 1
                if j == 0:
                    original_features.append(feats)
                    original_labels.append(1)

        print(f"  [MISPLACED] {fname}: {aug_count} augmented samples")
        group_idx += 1

    if not all_features:
        print("❌ No features extracted!")
        return

    n_correct = sum(1 for l in all_labels if l == 0)
    n_misplaced = sum(1 for l in all_labels if l == 1)
    print(f"\n  Total augmented samples: {len(all_features)} "
          f"({n_correct} correct, {n_misplaced} misplaced)")

    # Build arrays
    feature_keys = sorted(all_features[0].keys())
    X = np.array([[f[k] for k in feature_keys] for f in all_features])
    y = np.array(all_labels)
    groups = np.array(all_groups)

    X_orig = np.array([[f[k] for k in feature_keys] for f in original_features])
    y_orig = np.array(original_labels)

    print(f"  Features ({len(feature_keys)}): {', '.join(feature_keys)}")

    # Step 4: Train SVM classifier
    print("\n🧮 Step 4: Training SVM classifier (RBF kernel)...")

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import LeaveOneGroupOut

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True))
    ])

    # Step 5: Leave-one-group-out cross-validation
    print("\n📊 Step 5: Leave-one-group-out cross-validation...")
    logo = LeaveOneGroupOut()

    cv_correct = 0
    cv_total = 0
    fold = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)

        # Use majority vote across augmented versions of this image
        vote = Counter(test_pred).most_common(1)[0][0]
        actual = y_test[0]  # all same label within a group

        is_correct = vote == actual
        cv_correct += int(is_correct)
        cv_total += 1
        label_str = "CORRECT" if actual == 0 else "MISPLACED"
        pred_str = "CORRECT" if vote == 0 else "MISPLACED"
        status = "✅" if is_correct else "❌"
        print(f"  Fold {fold:2d}: actual={label_str:10s} predicted={pred_str:10s} {status}")

    cv_accuracy = cv_correct / max(cv_total, 1) * 100
    print(f"\n  📈 Cross-validation accuracy: {cv_accuracy:.1f}% ({cv_correct}/{cv_total})")

    # Step 6: Train final model on all data
    print("\n🔧 Step 6: Training final model on all data...")
    pipeline.fit(X, y)

    train_pred = pipeline.predict(X)
    train_accuracy = float(np.mean(train_pred == y) * 100)
    print(f"  ✅ Training accuracy: {train_accuracy:.1f}%")

    orig_pred = pipeline.predict(X_orig)
    orig_accuracy = float(np.mean(orig_pred == y_orig) * 100)
    print(f"  ✅ Original image accuracy: {orig_accuracy:.1f}%")

    # Step 7: Feature importance analysis
    print("\n📏 Step 7: Feature importance analysis...")

    feature_importance = {}
    for i, key in enumerate(feature_keys):
        correct_vals = X[y == 0, i]
        misplaced_vals = X[y == 1, i]
        separation = abs(correct_vals.mean() - misplaced_vals.mean())
        pooled_std = np.sqrt((correct_vals.std()**2 + misplaced_vals.std()**2) / 2) + 1e-8
        fisher_score = separation / pooled_std
        feature_importance[key] = {
            "fisher_score": float(fisher_score),
            "correct_mean": float(correct_vals.mean()),
            "correct_std": float(correct_vals.std()),
            "misplaced_mean": float(misplaced_vals.mean()),
            "misplaced_std": float(misplaced_vals.std()),
            "threshold": float((correct_vals.mean() + misplaced_vals.mean()) / 2),
            "direction": "higher_is_misplaced" if misplaced_vals.mean() > correct_vals.mean() else "lower_is_misplaced"
        }

    sorted_features = sorted(feature_importance.items(),
                             key=lambda x: x[1]["fisher_score"], reverse=True)

    print("\n  Top discriminative features:")
    for fname, finfo in sorted_features[:8]:
        print(f"    {fname}: score={finfo['fisher_score']:.3f} "
              f"(correct={finfo['correct_mean']:.4f}, misplaced={finfo['misplaced_mean']:.4f})")

    # Step 8: Save models
    print("\n💾 Step 8: Saving trained models...")

    # Save SVM pipeline
    os.makedirs(os.path.dirname(SVM_MODEL_PATH), exist_ok=True)
    with open(SVM_MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  ✅ SVM model saved: {SVM_MODEL_PATH}")

    # Compute reference stats for profile
    def avg_features(feature_list):
        keys = feature_list[0].keys()
        return {k: float(np.mean([f[k] for f in feature_list])) for k in keys}

    def std_features(feature_list):
        keys = feature_list[0].keys()
        return {k: float(np.std([f[k] for f in feature_list])) for k in keys}

    correct_feats_only = [f for f, l in zip(all_features, all_labels) if l == 0]
    misplaced_feats_only = [f for f, l in zip(all_features, all_labels) if l == 1]

    # Compute centroids for fallback classifier
    scaler = pipeline.named_steps['scaler']
    X_norm = scaler.transform(X)
    correct_centroid = X_norm[y == 0].mean(axis=0)
    misplaced_centroid = X_norm[y == 1].mean(axis=0)
    decision_boundary = (correct_centroid + misplaced_centroid) / 2
    decision_direction = misplaced_centroid - correct_centroid
    decision_direction = decision_direction / (np.linalg.norm(decision_direction) + 1e-8)

    profile = {
        "version": "2.0",
        "trained_at": datetime.now().isoformat(),
        "training_samples": {
            "correct": len(correct_files),
            "misplaced": len(misplaced_files),
            "augmented_total": len(all_features)
        },
        "training_accuracy": train_accuracy,
        "cv_accuracy": cv_accuracy,
        "original_accuracy": orig_accuracy,
        "feature_keys": feature_keys,
        "normalization": {
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist()
        },
        "classifier": {
            "type": "svm_rbf",
            "svm_model_path": "scene_classifier.pkl",
            "fallback_type": "centroid_linear",
            "correct_centroid": correct_centroid.tolist(),
            "misplaced_centroid": misplaced_centroid.tolist(),
            "decision_boundary": decision_boundary.tolist(),
            "decision_direction": decision_direction.tolist()
        },
        "reference_profiles": {
            "correct": {
                "avg": avg_features(correct_feats_only),
                "std": std_features(correct_feats_only),
            },
            "misplaced": {
                "avg": avg_features(misplaced_feats_only),
                "std": std_features(misplaced_feats_only),
            }
        },
        "feature_importance": {k: v for k, v in sorted_features},
        "thresholds": {
            k: {
                "value": v["threshold"],
                "direction": v["direction"],
                "score": v["fisher_score"]
            }
            for k, v in sorted_features if v["fisher_score"] > 0.1
        }
    }

    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"  ✅ Profile saved: {PROFILE_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Original images: {len(correct_files)} correct + {len(misplaced_files)} misplaced")
    print(f"  Augmented samples: {len(all_features)}")
    print(f"  Features: {len(feature_keys)}")
    print(f"  Cross-validation accuracy: {cv_accuracy:.1f}%")
    print(f"  Training accuracy: {train_accuracy:.1f}%")
    print(f"  Original image accuracy: {orig_accuracy:.1f}%")
    print(f"\n  Models saved:")
    print(f"    {PROFILE_PATH}")
    print(f"    {SVM_MODEL_PATH}")
    print(f"\n  The analyzer now uses the trained SVM classifier")
    print(f"  with column-deviation heuristics for accurate detection! 🎯")


if __name__ == "__main__":
    train()
