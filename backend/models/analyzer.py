"""
ML Chair Detection & Alignment Analysis Engine.
Uses a local YOLO model to analyze chair arrangement in lab environments.
"""

import cv2
import numpy as np
import logging
import os
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace

logger = logging.getLogger("backend.analyzer")

TRAINED_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "trained_profile.json")
SVM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene_classifier.pkl")


@dataclass
class ChairAnalysis:
    """Analysis result for a single chair."""
    chair_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    is_properly_arranged: bool
    nearest_desk_id: Optional[int]
    distance_to_desk: Optional[float]
    alignment_score: float
    issues: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    total_chairs: int
    total_desks: int
    correct_chairs: int
    misplaced_chairs: int
    accuracy: float
    avg_confidence: float
    chairs: List[Dict[str, Any]]
    desks: List[Dict[str, Any]]
    image_width: int
    image_height: int
    scene_classification: str = "unknown"
    scene_confidence: float = 0.0
    ai_description: str = ""
    ai_provider: str = "unknown"
    ai_model: str = ""


class ChairAnalyzer:
    """
    Local-only chair arrangement analyzer using a YOLO model.
    No external LLM or cloud vision API is required.
    """

    def __init__(self):
        self.provider = "local"
        self.local_model_path = self._find_local_model_path()
        self.model_name = "local_yolo"
        self.is_ready_flag = False
        self.trained_profile = self._load_trained_profile()
        self.svm_model = self._load_svm_model()

        if self.local_model_path:
            self.is_ready_flag = True
            logger.info("✅ Local YOLO model available: %s", os.path.basename(self.local_model_path))
        else:
            logger.warning("⚠️ Local YOLO model not found. Detection will not work.")

        if self.trained_profile:
            logger.info("✅ Trained profile loaded: %s", TRAINED_PROFILE_PATH)
        else:
            logger.warning("⚠️ Trained profile not found or invalid. Using heuristic flags only.")

        if self.svm_model:
            logger.info("✅ SVM scene classifier loaded: %s", SVM_MODEL_PATH)

        # YOLO model (lazy-loaded)
        self._yolo_model = None
        try:
            from ultralytics import YOLO  # type: ignore
            self._YOLO_CLASS = YOLO
        except Exception:
            self._YOLO_CLASS = None

    @property
    def is_ready(self) -> bool:
        return self.is_ready_flag

    def detect_objects(self, image: np.ndarray) -> List[Any]:
        """Lightweight object detector that uses Ultralyics YOLO.

        Returns an empty list when no detector is configured. This keeps
        the analyzer usable in environments without a local detector.
        """
        # Prefer ultralytics YOLO if available and a model file exists.
        if self._YOLO_CLASS is None or not self.local_model_path:
            return []

        model_path = self.local_model_path

        # Lazy load model
        if self._yolo_model is None:
            try:
                self._yolo_model = self._YOLO_CLASS(model_path)
            except Exception:
                return []

        # Run inference
        try:
            results = self._yolo_model.predict(source=image, conf=0.15, verbose=False)
        except TypeError:
            # older API fallback
            results = self._yolo_model(image)

        detections: List[Any] = []
        # results may be a list of Result objects (one per image)
        for res in results:
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, 'xyxy', None)
            cls = getattr(boxes, 'cls', None)
            conf = getattr(boxes, 'conf', None)
            # Some ultralytics versions use numpy arrays
            if xyxy is None:
                continue

            try:
                arr_xyxy = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else (xyxy.numpy() if hasattr(xyxy, 'numpy') else xyxy)
            except Exception:
                arr_xyxy = xyxy

            try:
                arr_cls = cls.cpu().numpy() if hasattr(cls, 'cpu') else (cls.numpy() if hasattr(cls, 'numpy') else cls)
            except Exception:
                arr_cls = cls

            try:
                arr_conf = conf.cpu().numpy() if hasattr(conf, 'cpu') else (conf.numpy() if hasattr(conf, 'numpy') else conf)
            except Exception:
                arr_conf = conf

            for i, box in enumerate(arr_xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                class_id = int(arr_cls[i]) if arr_cls is not None else -1
                confidence = float(arr_conf[i]) if arr_conf is not None else 0.0

                det = SimpleNamespace()
                det.class_id = class_id
                det.bbox = (x1, y1, x2, y2)
                det.center = center
                det.width = width
                det.height = height
                det.confidence = confidence
                detections.append(det)

        return detections

    def _find_local_model_path(self) -> Optional[str]:
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt'),
            os.path.join(os.getcwd(), 'backend', 'yolov8n.pt'),
            os.path.join(os.getcwd(), 'yolov8n.pt'),
        ]
        for p in candidates:
            p = os.path.normpath(p)
            if os.path.exists(p):
                return p
        return None

    def _load_trained_profile(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(TRAINED_PROFILE_PATH):
            return None

        try:
            with open(TRAINED_PROFILE_PATH, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            return profile
        except Exception as exc:
            logger.warning("Failed to load trained profile: %s", exc)
            return None

    def _load_svm_model(self):
        """Load the trained SVM classifier from pickle file."""
        if not os.path.exists(SVM_MODEL_PATH):
            return None
        try:
            with open(SVM_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as exc:
            logger.warning("Failed to load SVM model: %s", exc)
            return None

    def _normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        if not self.trained_profile:
            return np.array([])

        keys = self.trained_profile.get('feature_keys', [])
        mean = np.array(self.trained_profile.get('normalization', {}).get('mean', []), dtype=float)
        std = np.array(self.trained_profile.get('normalization', {}).get('std', []), dtype=float)
        values = np.array([features.get(k, 0.0) for k in keys], dtype=float)
        if len(values) != len(mean) or len(values) != len(std):
            return values

        return (values - mean) / (std + 1e-8)

    def _classify_scene(self, features: Dict[str, float]) -> Tuple[str, float]:
        import math

        # Sanitize features
        safe_features = {}
        for k, v in features.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                safe_features[k] = 0.0
            else:
                safe_features[k] = float(v)

        # Try SVM model first (more accurate)
        if self.svm_model is not None and self.trained_profile:
            try:
                keys = self.trained_profile.get('feature_keys', [])
                x = np.array([[safe_features.get(k, 0.0) for k in keys]])

                prediction = self.svm_model.predict(x)[0]

                if hasattr(self.svm_model, 'predict_proba'):
                    probabilities = self.svm_model.predict_proba(x)[0]
                    confidence = float(max(probabilities)) * 100.0
                else:
                    confidence = 75.0

                scene = "misplaced_arrangement" if prediction == 1 else "correct_arrangement"
                return scene, confidence
            except Exception as e:
                logger.warning("SVM classification failed, falling back to centroid: %s", e)

        # Fall back to centroid-based classifier
        if not self.trained_profile:
            return "local_fallback", 0.0

        x_norm = self._normalize_features(safe_features)
        if x_norm.size == 0:
            return "local_fallback", 0.0

        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

        classifier = self.trained_profile.get('classifier', {})
        direction = np.array(classifier.get('decision_direction', []), dtype=float)
        boundary = np.array(classifier.get('decision_boundary', []), dtype=float)
        if direction.size != x_norm.size or boundary.size != x_norm.size:
            return "local_fallback", 0.0

        score = float(np.dot(x_norm - boundary, direction))
        if math.isnan(score) or math.isinf(score):
            score = 0.0

        scene = "misplaced_arrangement" if score > 0 else "correct_arrangement"
        confidence = min(max((score + 1.0) / 2.0, 0.0), 1.0)
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.0
        return scene, confidence * 100.0

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            f = float(v)
        except Exception:
            return default
        if np.isnan(f) or np.isinf(f):
            return default
        return f

    def _find_nearest_desk(self, chair: Any, desks: List[Any]) -> Tuple[Optional[Any], float, float]:
        best_desk = None
        best_dist = float('inf')
        best_overlap = 0.0
        x1, y1, x2, y2 = chair.bbox
        chair_area = max(1, (x2 - x1) * (y2 - y1))

        for desk in desks:
            dx1, dy1, dx2, dy2 = desk.bbox
            ix1 = max(x1, dx1)
            iy1 = max(y1, dy1)
            ix2 = min(x2, dx2)
            iy2 = min(y2, dy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = chair_area + max(1, (dx2 - dx1) * (dy2 - dy1)) - inter
            overlap_ratio = inter / max(union, 1)
            center_dist = np.sqrt((chair.center[0] - desk.center[0])**2 + (chair.center[1] - desk.center[1])**2)
            if overlap_ratio > best_overlap or (overlap_ratio == best_overlap and center_dist < best_dist):
                best_overlap = overlap_ratio
                best_dist = center_dist
                best_desk = desk

        return best_desk, best_dist, best_overlap

    def has_local_fallback(self) -> bool:
        return self._YOLO_CLASS is not None and self._find_local_model_path() is not None

    def _local_fallback_analysis(self, image: np.ndarray) -> AnalysisResult:
        objects = self.detect_objects(image)
        chairs_raw = [o for o in objects if getattr(o, 'class_id', None) == 56]
        desks = [o for o in objects if getattr(o, 'class_id', None) == 60]

        # Filter overlapping chair detections (e.g., double boxes on the same chair)
        chairs = []
        for c in chairs_raw:
            overlap = False
            for existing in chairs:
                if np.hypot(c.center[0] - existing.center[0], c.center[1] - existing.center[1]) < 40:
                    overlap = True
                    break
            if not overlap:
                chairs.append(c)

        # Sort chairs: Right row first, then Left row. Within row: Front (bottom, max y) to Back (top, min y).
        n_right = 0
        midpoint = 0.0

        if chairs:
            min_x = min(c.center[0] for c in chairs)
            max_x = max(c.center[0] for c in chairs)
            midpoint = (min_x + max_x) / 2.0

            right_chairs = [c for c in chairs if c.center[0] > midpoint]
            left_chairs = [c for c in chairs if c.center[0] <= midpoint]

            right_chairs.sort(key=lambda c: c.center[1], reverse=True)
            left_chairs.sort(key=lambda c: c.center[1], reverse=True)

            chairs = right_chairs + left_chairs
            n_right = len(right_chairs)

        total_chairs = len(chairs)

        scene_features = self._build_scene_features(image, chairs, desks)
        scene_classification, scene_confidence = self._classify_scene(scene_features)

        # Dynamic threshold based on scene classification
        # If the scene is overall correct, we need a huge deviation to flag a chair.
        # If it's misplaced, we use a tighter threshold to catch the offending chairs.
        aisle_threshold = 50 if scene_classification == "misplaced_arrangement" else 300

        # Robust edge-anchored regression to detect pulled-out chairs
        pulled_out_set = set()  # indices (0-based) of chairs flagged as pulled out
        if total_chairs >= 4:
            for col_chairs, col_start in [
                (chairs[:n_right], 0),
                (chairs[n_right:], n_right)
            ]:
                if len(col_chairs) < 3:
                    continue

                xs = np.array([c.center[0] for c in col_chairs], dtype=float)
                ys = np.array([c.center[1] for c in col_chairs], dtype=float)

                # Step 1: Fit initial line to all chairs to capture the general slant
                coeffs_init = np.polyfit(ys, xs, 1)
                pred_x_init = np.polyval(coeffs_init, ys)

                # Step 2: Compute deviations (both columns face left, walkway is to the right = larger X)
                devs = xs - pred_x_init

                # Step 3: Select tucked-in chairs (smallest deviations)
                tucked_in_indices = np.argsort(devs)[:3]
                edge_chairs = [col_chairs[idx] for idx in tucked_in_indices]
                edge_xs = np.array([c.center[0] for c in edge_chairs], dtype=float)
                edge_ys = np.array([c.center[1] for c in edge_chairs], dtype=float)

                y_spread = np.max(edge_ys) - np.min(edge_ys) if len(edge_ys) > 0 else 0

                if y_spread < 50 or len(edge_chairs) < 2:
                    # Not enough vertical spread to fit a reliable line, use vertical median
                    median_x = float(np.median(edge_xs))
                    coeffs = np.array([0.0, median_x])
                else:
                    coeffs = np.polyfit(edge_ys, edge_xs, 1)

                # Evaluate all chairs in the column against the robust edge line
                for j, c in enumerate(col_chairs):
                    pred_x = np.polyval(coeffs, c.center[1])
                    aisle_res = c.center[0] - pred_x
                        
                    # If chair is beyond threshold into the aisle relative to the edge anchor line, flag it
                    if aisle_res > aisle_threshold:
                        pulled_out_set.add(col_start + j)

        chair_analyses = []

        for i, chair in enumerate(chairs, start=1):
            nearest_desk, distance_to_desk, overlap_ratio = self._find_nearest_desk(chair, desks)
            issues = []
            properly_arranged = True

            # Per-chair classification: use the pre-computed regression outlier set
            if (i - 1) in pulled_out_set:
                issues.append("Pulled out")
                properly_arranged = False

            # Alignment score
            alignment_score = 100.0
            if "Pulled out" in issues:
                alignment_score -= 35.0
            alignment_score = max(0.0, min(100.0, alignment_score))

            is_dist_invalid = distance_to_desk == float('inf') or np.isinf(distance_to_desk) or np.isnan(distance_to_desk)
            chair_analyses.append(ChairAnalysis(
                chair_id=i,
                bbox=getattr(chair, 'bbox', (0, 0, 0, 0)),
                center=getattr(chair, 'center', (0, 0)),
                confidence=self._safe_float(getattr(chair, 'confidence', 0.0)),
                is_properly_arranged=properly_arranged,
                nearest_desk_id=getattr(nearest_desk, 'class_id', None),
                distance_to_desk=None if is_dist_invalid else float(distance_to_desk),
                alignment_score=alignment_score,
                issues=issues
            ))

        misplaced = len([c for c in chair_analyses if not c.is_properly_arranged])
        avg_confidence = self._safe_float(float(np.mean([c.confidence for c in chair_analyses]))) if chair_analyses else 0.0

        return AnalysisResult(
            total_chairs=total_chairs,
            total_desks=len(desks),
            correct_chairs=total_chairs - misplaced,
            misplaced_chairs=misplaced,
            accuracy=self._safe_float(100.0 * (total_chairs - misplaced) / total_chairs) if total_chairs > 0 else 0.0,
            avg_confidence=self._safe_float(avg_confidence),
            chairs=[asdict(c) for c in chair_analyses],
            desks=[{'bbox': getattr(d, 'bbox', (0,0,0,0)), 'center': getattr(d, 'center', (0,0)), 'confidence': self._safe_float(getattr(d, 'confidence', 0.0))} for d in desks],
            image_width=int(image.shape[1]) if image is not None else 0,
            image_height=int(image.shape[0]) if image is not None else 0,
            scene_classification=scene_classification,
            scene_confidence=self._safe_float(scene_confidence),
            ai_description=f"ML model detected {total_chairs} chair(s) and flagged {misplaced} as misplaced (pulled out or tilted).",
            ai_provider='local_yolo',
            ai_model=os.path.basename(self.local_model_path or 'yolov8n.pt')
        )

    def _build_scene_features(self, image: np.ndarray, chairs: List[Any], desks: List[Any]) -> Dict[str, float]:
        h, w = image.shape[:2]
        img_area = h * w
        img_diagonal = np.sqrt(w**2 + h**2)

        chair_positions_x = [c.center[0] / w for c in chairs] if chairs else [0.0]
        chair_positions_y = [c.center[1] / h for c in chairs] if chairs else [0.0]
        chair_aspect_ratios = [c.height / max(c.width, 1) for c in chairs] if chairs else [0.0]

        spacings = []
        if len(chairs) >= 2:
            sorted_chairs = sorted(chairs, key=lambda c: c.center[1])
            for i in range(1, len(sorted_chairs)):
                dx = sorted_chairs[i].center[0] - sorted_chairs[i-1].center[0]
                dy = sorted_chairs[i].center[1] - sorted_chairs[i-1].center[1]
                spacings.append(np.sqrt(dx*dx + dy*dy) / img_diagonal)

        # Column analysis
        max_col_dev = 0.0
        aisle_ratio = 0.0
        if len(chairs) >= 2:
            min_cx = min(c.center[0] for c in chairs)
            max_cx = max(c.center[0] for c in chairs)
            mid_cx = (min_cx + max_cx) / 2.0
            col_sep = max(max_cx - min_cx, 1.0)

            right = [c for c in chairs if c.center[0] > mid_cx]
            left = [c for c in chairs if c.center[0] <= mid_cx]

            x_devs = []
            if len(right) >= 2:
                r_med = float(np.median([c.center[0] for c in right]))
                for c in right:
                    x_devs.append(max(0, c.center[0] - r_med) / col_sep)
            if len(left) >= 2:
                l_med = float(np.median([c.center[0] for c in left]))
                for c in left:
                    x_devs.append(max(0, c.center[0] - l_med) / col_sep)

            if x_devs:
                max_col_dev = float(max(x_devs))
                aisle_ratio = float(sum(1 for d in x_devs if d > 0.08) / len(chairs))

        # Visual features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lower_floor = np.array([0, 0, 150])
        upper_floor = np.array([180, 60, 255])
        floor_mask = cv2.inRange(hsv, lower_floor, upper_floor)
        lower_half = floor_mask[h//2:, :]
        floor_exposure = float(np.sum(lower_half > 0) / max(lower_half.size, 1))

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges[h//2:, :] > 0) / max(edges[h//2:, :].size, 1))

        lower_blue = np.array([90, 40, 80])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_exposure = float(np.sum(blue_mask > 0) / max(blue_mask.size, 1))

        # Chair coverage
        total_area = sum(c.width * c.height for c in chairs) if chairs else 0
        chair_coverage = float(total_area / img_area) if chairs else 0.0

        return {
            "aisle_chair_ratio": aisle_ratio,
            "aspect_variance": float(np.var(chair_aspect_ratios)) if len(chair_aspect_ratios) > 1 else 0.0,
            "avg_aspect_ratio": float(np.mean(chair_aspect_ratios)) if chair_aspect_ratios else 0.0,
            "avg_spacing": float(np.mean(spacings)) if spacings else 0.0,
            "blue_desk_exposure": blue_exposure,
            "chair_coverage_ratio": chair_coverage,
            "edge_density": edge_density,
            "floor_exposure": floor_exposure,
            "max_column_x_deviation": max_col_dev,
            "std_chair_x": float(np.std(chair_positions_x)) if len(chair_positions_x) > 1 else 0.0,
            "std_chair_y": float(np.std(chair_positions_y)) if len(chair_positions_y) > 1 else 0.0,
            "std_spacing": float(np.std(spacings)) if len(spacings) > 1 else 0.0,
            "x_variance": float(np.var(chair_positions_x)) if len(chair_positions_x) > 1 else 0.0,
            "y_variance": float(np.var(chair_positions_y)) if len(chair_positions_y) > 1 else 0.0,
        }

    def _estimate_tilt_angle(self, obj: Any, image: np.ndarray) -> float:
        x1, y1, x2, y2 = obj.bbox
        roi = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return 0.0

        rect = cv2.minAreaRect(largest)
        angle = rect[-1]
        if rect[1][0] < rect[1][1]:
            angle = 90.0 + angle
        angle = abs(angle)
        if angle > 90.0:
            angle = 180.0 - angle
        return angle

    def _is_chair_pulled_out_of_row(self, obj: Any, chairs: List[Any]) -> bool:
        x_positions = [c.center[0] for c in chairs]
        if len(x_positions) < 2:
            return False
        avg_x = sum(x_positions) / len(x_positions)
        return abs(obj.center[0] - avg_x) > max(1.0, np.std(x_positions) * 2.0)

    def analyze_arrangement(self, image: np.ndarray) -> AnalysisResult:
        """Full analysis pipeline using local YOLO detection."""
        h, w = image.shape[:2]

        if not self.has_local_fallback():
            logger.error("Local YOLO model not available for analysis.")
            return AnalysisResult(
                total_chairs=0, total_desks=0, correct_chairs=0, misplaced_chairs=0,
                accuracy=0.0, avg_confidence=0.0, chairs=[], desks=[],
                image_width=w, image_height=h, scene_classification="local_model_missing",
                ai_provider="local_yolo",
                ai_model="none"
            )

        result = self._local_fallback_analysis(image)
        logger.info(
            "AnalysisResult generated: provider=%s model=%s total_chairs=%d misplaced=%d accuracy=%.1f scene=%s",
            result.ai_provider,
            result.ai_model,
            result.total_chairs,
            result.misplaced_chairs,
            result.accuracy,
            result.scene_classification
        )
        return result

    def annotate_image(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """
        Draw a clean summary overlay on the original image.
        No bounding boxes — just the stats and AI description.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate overlay height based on content
        lines = []
        lines.append(f"Total Chairs: {result.total_chairs}")
        lines.append(f"Properly Arranged: {result.correct_chairs}  |  Misplaced: {result.misplaced_chairs}")

        if result.accuracy >= 80:
            acc_text = f"Arrangement Accuracy: {result.accuracy}%"
        else:
            acc_text = f"Arrangement Accuracy: {result.accuracy}%"
        lines.append(acc_text)

        # Add misplaced chair details
        misplaced_details = []
        for chair in result.chairs:
            if not chair["is_properly_arranged"] and chair["issues"]:
                misplaced_details.append(f"  Chair #{chair['chair_id']}: {', '.join(chair['issues'])}")

        overlay_height = 100 + len(misplaced_details) * 22 + 25
        overlay_width = min(w, 600)

        # Draw semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

        # Draw summary text
        y_pos = 24
        cv2.putText(annotated, lines[0], (10, y_pos), font, 0.62, (255, 255, 255), 1)
        y_pos += 24
        cv2.putText(annotated, lines[1], (10, y_pos), font, 0.55, (255, 255, 255), 1)
        y_pos += 28

        acc_color = (0, 220, 80) if result.accuracy >= 80 else (0, 180, 255) if result.accuracy >= 50 else (0, 60, 230)
        cv2.putText(annotated, acc_text, (10, y_pos), font, 0.72, acc_color, 2)
        y_pos += 24

        cv2.putText(annotated, "ML: Local YOLO detection", (10, y_pos), font, 0.42, (160, 160, 160), 1)
        y_pos += 20

        # Draw misplaced chair details
        if misplaced_details:
            cv2.putText(annotated, "Misplaced:", (10, y_pos), font, 0.5, (0, 100, 255), 1)
            y_pos += 20
            for detail in misplaced_details:
                cv2.putText(annotated, detail, (10, y_pos), font, 0.42, (0, 130, 255), 1)
                y_pos += 22

        return annotated

    def generate_heatmap(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """Generate a simple heatmap overlay. Since we don't have bboxes, just return the annotated image."""
        # Without precise bounding boxes, return a copy of the image with status overlay
        return self.annotate_image(image, result)


# Singleton
_analyzer_instance = None


def get_analyzer() -> ChairAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ChairAnalyzer()
    return _analyzer_instance


def reload_analyzer():
    global _analyzer_instance
    _analyzer_instance = ChairAnalyzer()
    return _analyzer_instance
