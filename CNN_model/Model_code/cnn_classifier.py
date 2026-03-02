"""Lightweight CNN-style victim classifier implemented in NumPy."""

from __future__ import annotations

from dataclasses import dataclass
import os
import math
import numpy as np


@dataclass(frozen=True)
class ClassPrediction:
    label: str
    confidence: float


class TinyCNNClassifier:
    """Small deterministic conv net for on-controller inference.

    The network is intentionally compact:
    conv(4 filters) -> relu -> global average pool -> linear classifier.
    """

    def __init__(
        self,
        input_size: int = 24,
        labels: tuple[str, ...] = ("H", "S", "U"),
        weights_path: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.labels = tuple(labels)
        self.pool_size = 4
        self.model_version = "legacy"
        self.kernels = np.array(
            [
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                [[2, -1, 0], [-1, 0, 1], [0, 1, -2]],
                [[0, -1, 2], [1, 0, -1], [-2, 1, 0]],
            ],
            dtype=np.float32,
        )
        feature_dim = self.kernels.shape[0] * self.pool_size * self.pool_size
        self.weights = np.zeros((len(self.labels), feature_dim), dtype=np.float32)
        for idx in range(min(len(self.labels), feature_dim)):
            self.weights[idx, idx] = 1.0
        self.bias = np.zeros((len(self.labels),), dtype=np.float32)
        self.conv1_w: np.ndarray | None = None
        self.conv1_b: np.ndarray | None = None
        self.conv2_w: np.ndarray | None = None
        self.conv2_b: np.ndarray | None = None
        self.fc_w: np.ndarray | None = None
        self.fc_b: np.ndarray | None = None
        if weights_path:
            self._load_weights(weights_path)

    def predict(self, rgb_crop: np.ndarray) -> ClassPrediction:
        x = self._preprocess(rgb_crop)
        if x is None:
            return ClassPrediction(label=self.labels[0], confidence=0.0)
        if self.model_version == "v2":
            features = self._extract_features_v2(x)
            logits = np.dot(self.fc_w, features) + self.fc_b
        else:
            features = self._extract_features(x)
            logits = np.dot(self.weights, features) + self.bias
        probs = self._softmax(logits)
        idx = int(np.argmax(probs))
        return ClassPrediction(label=self.labels[idx], confidence=float(probs[idx]))

    def _load_weights(self, weights_path: str) -> None:
        if not os.path.exists(weights_path):
            print(f"event cnn_weights_missing path={weights_path}")
            return
        data = np.load(weights_path, allow_pickle=True)
        labels = tuple(str(item) for item in data["labels"].tolist())
        input_size = int(data["input_size"])
        if "model_version" in data and str(data["model_version"]) == "v2":
            self._load_v2_weights(data, weights_path, labels, input_size)
            return

        kernels = np.array(data["kernels"], dtype=np.float32)
        weights = np.array(data["weights"], dtype=np.float32)
        bias = np.array(data["bias"], dtype=np.float32)
        pool_size = int(data.get("pool_size", 4))

        if kernels.ndim != 3 or kernels.shape[1:] != (3, 3):
            raise ValueError(f"Invalid kernels shape in {weights_path}: {kernels.shape}")
        channels = kernels.shape[0]
        feature_dim = channels * pool_size * pool_size
        # Backward compatibility with older exported weights using global pooling.
        if weights.shape == (len(labels), channels):
            pool_size = 1
            feature_dim = channels
        if weights.shape != (len(labels), feature_dim):
            raise ValueError(f"Invalid weights shape in {weights_path}: {weights.shape}")
        if bias.shape != (len(labels),):
            raise ValueError(f"Invalid bias shape in {weights_path}: {bias.shape}")
        if input_size <= 0:
            raise ValueError(f"Invalid input_size in {weights_path}: {input_size}")

        self.kernels = kernels
        self.weights = weights
        self.bias = bias
        self.labels = labels
        self.input_size = input_size
        self.pool_size = pool_size
        self.model_version = "legacy"
        print(f"event cnn_weights_loaded path={weights_path} labels={self.labels}")

    def _load_v2_weights(self, data, weights_path: str, labels: tuple[str, ...], input_size: int) -> None:
        conv1_w = np.array(data["conv1_w"], dtype=np.float32)
        conv1_b = np.array(data["conv1_b"], dtype=np.float32)
        conv2_w = np.array(data["conv2_w"], dtype=np.float32)
        conv2_b = np.array(data["conv2_b"], dtype=np.float32)
        fc_w = np.array(data["fc_w"], dtype=np.float32)
        fc_b = np.array(data["fc_b"], dtype=np.float32)
        pool_size = int(data.get("pool_size", 2))

        if conv1_w.ndim != 4 or conv1_w.shape[1] != 1 or conv1_w.shape[2:] != (3, 3):
            raise ValueError(f"Invalid conv1_w shape in {weights_path}: {conv1_w.shape}")
        if conv2_w.ndim != 4 or conv2_w.shape[2:] != (3, 3):
            raise ValueError(f"Invalid conv2_w shape in {weights_path}: {conv2_w.shape}")
        if conv2_w.shape[1] != conv1_w.shape[0]:
            raise ValueError(
                f"conv channel mismatch in {weights_path}: conv1_out={conv1_w.shape[0]} conv2_in={conv2_w.shape[1]}"
            )
        expected_fc = conv2_w.shape[0] * pool_size * pool_size
        if fc_w.shape != (len(labels), expected_fc):
            raise ValueError(f"Invalid fc_w shape in {weights_path}: {fc_w.shape}")
        if fc_b.shape != (len(labels),):
            raise ValueError(f"Invalid fc_b shape in {weights_path}: {fc_b.shape}")
        if conv1_b.shape != (conv1_w.shape[0],):
            raise ValueError(f"Invalid conv1_b shape in {weights_path}: {conv1_b.shape}")
        if conv2_b.shape != (conv2_w.shape[0],):
            raise ValueError(f"Invalid conv2_b shape in {weights_path}: {conv2_b.shape}")

        self.conv1_w = conv1_w
        self.conv1_b = conv1_b
        self.conv2_w = conv2_w
        self.conv2_b = conv2_b
        self.fc_w = fc_w
        self.fc_b = fc_b
        self.labels = labels
        self.input_size = input_size
        self.pool_size = pool_size
        self.model_version = "v2"
        print(f"event cnn_weights_loaded path={weights_path} labels={self.labels} model=v2")

    def _preprocess(self, rgb_crop: np.ndarray) -> np.ndarray | None:
        if rgb_crop.size == 0:
            return None
        gray = (
            0.299 * rgb_crop[:, :, 0].astype(np.float32)
            + 0.587 * rgb_crop[:, :, 1].astype(np.float32)
            + 0.114 * rgb_crop[:, :, 2].astype(np.float32)
        )
        # Match training preprocessing: normalize, isolate likely glyph region, resize.
        lo = float(np.percentile(gray, 5))
        hi = float(np.percentile(gray, 95))
        if hi - lo > 1e-6:
            gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
        else:
            gray = np.clip(gray / 255.0, 0.0, 1.0)

        bright_mask = gray > 0.62
        dark_mask = gray < 0.38
        b_count = int(np.count_nonzero(bright_mask))
        d_count = int(np.count_nonzero(dark_mask))
        total = gray.size
        use_bright = 0 < b_count < total * 0.75
        use_dark = 0 < d_count < total * 0.75
        mask = None
        if use_bright and use_dark:
            mask = bright_mask if b_count <= d_count else dark_mask
        elif use_bright:
            mask = bright_mask
        elif use_dark:
            mask = dark_mask

        if mask is not None and np.count_nonzero(mask) >= 12:
            ys, xs = np.where(mask)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            pad_x = max(1, int((x1 - x0) * 0.15))
            pad_y = max(1, int((y1 - y0) * 0.15))
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(gray.shape[1], x1 + pad_x)
            y1 = min(gray.shape[0], y1 + pad_y)
            gray = gray[y0:y1, x0:x1]

        resized = self._resize_nn(gray, self.input_size, self.input_size)
        if resized is None:
            return None
        return resized.astype(np.float32)

    def _resize_nn(self, img: np.ndarray, out_h: int, out_w: int) -> np.ndarray | None:
        in_h, in_w = img.shape
        if in_h <= 0 or in_w <= 0:
            return None
        y_idx = (np.linspace(0, in_h - 1, out_h)).astype(int)
        x_idx = (np.linspace(0, in_w - 1, out_w)).astype(int)
        return img[y_idx][:, x_idx]

    def _extract_features(self, x: np.ndarray) -> np.ndarray:
        padded = np.pad(x, ((1, 1), (1, 1)), mode="edge")
        responses: list[np.ndarray] = []
        for kernel in self.kernels:
            conv = np.zeros_like(x)
            for ky in range(3):
                for kx in range(3):
                    conv += kernel[ky, kx] * padded[ky : ky + x.shape[0], kx : kx + x.shape[1]]
            conv = np.maximum(conv, 0.0)
            pooled = self._adaptive_avg_pool_2d(conv, self.pool_size, self.pool_size)
            responses.append(pooled.astype(np.float32))
        return np.concatenate([r.reshape(-1) for r in responses]).astype(np.float32)

    def _extract_features_v2(self, x: np.ndarray) -> np.ndarray:
        x3 = x[np.newaxis, :, :].astype(np.float32)  # (1,H,W)
        c1 = self._conv2d_multi(x3, self.conv1_w, self.conv1_b)
        c1 = np.maximum(c1, 0.0)
        p1 = self._max_pool2d(c1, 2, 2)
        c2 = self._conv2d_multi(p1, self.conv2_w, self.conv2_b)
        c2 = np.maximum(c2, 0.0)
        p2 = self._max_pool2d(c2, 2, 2)
        pooled = self._adaptive_avg_pool_multi(p2, self.pool_size, self.pool_size)
        return pooled.reshape(-1).astype(np.float32)

    @staticmethod
    def _conv2d_multi(
        x: np.ndarray, kernels: np.ndarray, bias: np.ndarray, padding: int = 1
    ) -> np.ndarray:
        # x: (C,H,W), kernels: (O,C,3,3), bias: (O,)
        c, h, w = x.shape
        out_c = kernels.shape[0]
        padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="edge")
        out = np.zeros((out_c, h, w), dtype=np.float32)
        for oc in range(out_c):
            acc = np.zeros((h, w), dtype=np.float32)
            for ic in range(c):
                k = kernels[oc, ic]
                pi = padded[ic]
                for ky in range(3):
                    for kx in range(3):
                        acc += k[ky, kx] * pi[ky : ky + h, kx : kx + w]
            acc += bias[oc]
            out[oc] = acc
        return out

    @staticmethod
    def _max_pool2d(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
        # x: (C,H,W)
        c, h, w = x.shape
        out_h = max(1, 1 + (h - kernel) // stride)
        out_w = max(1, 1 + (w - kernel) // stride)
        out = np.zeros((c, out_h, out_w), dtype=np.float32)
        for ch in range(c):
            for oy in range(out_h):
                y0 = oy * stride
                y1 = y0 + kernel
                for ox in range(out_w):
                    x0 = ox * stride
                    x1 = x0 + kernel
                    out[ch, oy, ox] = float(np.max(x[ch, y0:y1, x0:x1]))
        return out

    def _adaptive_avg_pool_multi(self, x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        # x: (C,H,W)
        c, h, w = x.shape
        out = np.zeros((c, out_h, out_w), dtype=np.float32)
        for ch in range(c):
            out[ch] = self._adaptive_avg_pool_2d(x[ch], out_h, out_w)
        return out

    @staticmethod
    def _adaptive_avg_pool_2d(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        h, w = img.shape
        result = np.zeros((out_h, out_w), dtype=np.float32)
        for oy in range(out_h):
            y0 = int(np.floor(oy * h / out_h))
            y1 = int(np.ceil((oy + 1) * h / out_h))
            for ox in range(out_w):
                x0 = int(np.floor(ox * w / out_w))
                x1 = int(np.ceil((ox + 1) * w / out_w))
                region = img[y0:y1, x0:x1]
                if region.size == 0:
                    result[oy, ox] = 0.0
                else:
                    result[oy, ox] = float(np.mean(region))
        return result

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        max_logit = float(np.max(logits))
        exp_vals = np.array([math.exp(float(v - max_logit)) for v in logits], dtype=np.float32)
        total = float(np.sum(exp_vals))
        if total <= 0.0:
            return np.full_like(exp_vals, 1.0 / len(exp_vals))
        return exp_vals / total
