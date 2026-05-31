import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("ml.segmentation")
settings = get_settings()

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except Exception:  # pragma: no cover - exercised when ML deps are not installed
    torch = None
    nn = None
    models = None


SEVERITY_MAP = {
    0: "no_damage",
    1: "minor",
    2: "major",
    3: "destroyed",
}
DEFAULT_CLASS_NAMES = list(SEVERITY_MAP.values())


class DamageSegmentationModel:
    """Damage inference facade.

    The repo's stored CV artifact is a ConvNeXt damage classifier. The backend
    still exposes GeoJSON-like damage features, so this class uses the
    classifier for tile-level severity when it can load it, and keeps a
    deterministic pseudo-segmentation fallback for demos/tests.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None
        self.model = None
        self.weights_loaded = False
        self.inference_mode = "mock"
        self.checkpoint_path: Optional[Path] = None
        self.class_names = DEFAULT_CLASS_NAMES.copy()
        self.model_metadata: Dict[str, Any] = {}
        self._load_weights()

    def _candidate_paths(self) -> List[Path]:
        raw = Path(settings.DAMAGE_MODEL_PATH)
        repo_root = Path(__file__).resolve().parents[3]
        backend_root = Path(__file__).resolve().parents[2]
        candidates = [
            raw,
            Path.cwd() / raw,
            backend_root / raw,
            repo_root / raw,
            repo_root / "models_CV" / "week12_convnext_tiny_gated_ce_best.pt.zip",
        ]
        return [p.resolve() for p in candidates]

    def _resolve_checkpoint_path(self) -> Optional[Path]:
        for path in self._candidate_paths():
            if path.exists() and path.is_file():
                return path
        return None

    def _load_weights(self):
        self.checkpoint_path = self._resolve_checkpoint_path()
        if self.checkpoint_path is None:
            logger.warning("No local damage checkpoint found; using mock inference", path=settings.DAMAGE_MODEL_PATH)
            return

        if torch is None or models is None or nn is None:
            logger.warning(
                "PyTorch/TorchVision not installed; using mock inference with checkpoint metadata",
                path=str(self.checkpoint_path),
            )
            self._load_metadata_without_torch()
            return

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.class_names = list(checkpoint.get("class_names", self._infer_class_names(state_dict)))
            self.model_metadata = {
                "checkpoint": str(self.checkpoint_path),
                "epoch": checkpoint.get("epoch"),
                "metrics": checkpoint.get("val_metrics", {}),
                "backbone": checkpoint.get("backbone", "convnext_tiny"),
                "fusion": checkpoint.get("fusion"),
                "loss_type": checkpoint.get("loss_type"),
            }
            self.model = self._build_convnext(len(self.class_names), state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.weights_loaded = True
            self.inference_mode = "convnext_classifier"
            logger.info(
                "Loaded local ConvNeXt damage checkpoint",
                path=str(self.checkpoint_path),
                classes=self.class_names,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load local damage checkpoint; using mock inference",
                path=str(self.checkpoint_path),
                error=str(exc),
            )

    def _load_metadata_without_torch(self):
        # Keep the public output aligned with the known artifact even when the
        # local Python environment has not installed torch yet.
        self.class_names = DEFAULT_CLASS_NAMES.copy()
        self.model_metadata = {
            "checkpoint": str(self.checkpoint_path),
            "backbone": "convnext_tiny",
            "fusion": "gated",
            "note": "Install torch/torchvision to run real checkpoint inference.",
        }

    def _infer_class_names(self, state_dict: Dict[str, Any]) -> List[str]:
        for key in ("classifier.bias", "encoder.classifier.2.bias"):
            tensor = state_dict.get(key)
            if tensor is not None and hasattr(tensor, "shape"):
                count = int(tensor.shape[0])
                return DEFAULT_CLASS_NAMES[:count]
        return DEFAULT_CLASS_NAMES.copy()

    def _build_convnext(self, num_classes: int, state_dict: Dict[str, Any]):
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

        encoder_state = {
            key.replace("encoder.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if encoder_state:
            missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        else:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if unexpected:
            logger.info("Ignored checkpoint layers not used by the backend classifier", count=len(unexpected))
        if missing:
            logger.info("ConvNeXt classifier initialized with partial checkpoint weights", missing=len(missing))
        return model

    def predict(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """
        image_array: HxWx3 numpy array (RGB, 0-255).
        Returns GeoJSON-like features with damage labels and confidence.
        """
        image_array = self._validate_image(image_array)
        if self.weights_loaded and self.model is not None:
            severity, confidence, probabilities = self._predict_tile_class(image_array)
            return [self._tile_feature(image_array, severity, confidence, probabilities)]
        return self._mock_predict(image_array)

    def _validate_image(self, image_array: np.ndarray) -> np.ndarray:
        if not isinstance(image_array, np.ndarray):
            raise TypeError("image_array must be a numpy array")
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        if image_array.ndim != 3 or image_array.shape[2] not in (3, 4):
            raise ValueError("image_array must have shape HxWx3 or HxWx4")
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        return image_array.astype(np.uint8, copy=False)

    def _predict_tile_class(self, image_array: np.ndarray):
        tensor = self._to_imagenet_tensor(image_array).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        class_idx = int(np.argmax(probs))
        severity = self.class_names[class_idx] if class_idx < len(self.class_names) else SEVERITY_MAP.get(class_idx, "unknown")
        return severity, round(float(probs[class_idx]), 4), probs.tolist()

    def _to_imagenet_tensor(self, image_array: np.ndarray):
        from PIL import Image

        image = Image.fromarray(image_array).resize((224, 224))
        arr = np.asarray(image).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def _tile_feature(self, image_array: np.ndarray, severity: str, confidence: float, probabilities: List[float]):
        h, w = image_array.shape[:2]
        coords = [[0, 0], [w, 0], [w, h], [0, h], [0, 0]]
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "severity": severity,
                "confidence": confidence,
                "area_sqm": float(w * h),
                "building_id": "tile_0",
                "inference_mode": self.inference_mode,
                "class_probabilities": {
                    self.class_names[i]: round(float(score), 4)
                    for i, score in enumerate(probabilities)
                    if i < len(self.class_names)
                },
                "model": self.model_metadata,
            },
        }

    def _mock_predict(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Deterministic fallback output for demos without installed weights."""
        h, w = image_array.shape[:2]
        digest = hashlib.sha256(image_array.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**32)
        rng = np.random.default_rng(seed)

        n_buildings = int(rng.integers(5, 20))
        min_x_margin = min(20, max(1, w // 8))
        min_y_margin = min(20, max(1, h // 8))
        features = []
        for i in range(n_buildings):
            cx = int(rng.integers(min_x_margin, max(min_x_margin + 1, w - min_x_margin)))
            cy = int(rng.integers(min_y_margin, max(min_y_margin + 1, h - min_y_margin)))
            bw = int(rng.integers(max(4, w // 40), max(5, w // 10)))
            bh = int(rng.integers(max(4, h // 40), max(5, h // 10)))
            severity = str(rng.choice(self.class_names, p=self._class_probabilities()))
            coords = [
                [max(0, cx - bw), max(0, cy - bh)],
                [min(w, cx + bw), max(0, cy - bh)],
                [min(w, cx + bw), min(h, cy + bh)],
                [max(0, cx - bw), min(h, cy + bh)],
                [max(0, cx - bw), max(0, cy - bh)],
            ]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "severity": severity,
                    "confidence": round(float(rng.uniform(0.65, 0.98)), 3),
                    "area_sqm": float(max(1, bw * bh) * 0.5),
                    "building_id": f"bldg_{i}",
                    "inference_mode": "mock_segmentation",
                    "model": self.model_metadata,
                },
            })
        return features

    def _class_probabilities(self):
        base = np.array([0.3, 0.3, 0.25, 0.15], dtype="float64")[:len(self.class_names)]
        return base / base.sum()


_segmentation_model = None


def get_segmentation_model() -> DamageSegmentationModel:
    global _segmentation_model
    if _segmentation_model is None:
        _segmentation_model = DamageSegmentationModel()
    return _segmentation_model
