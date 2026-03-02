"""Train a compact SHU classifier and export weights for TinyCNNClassifier.

Dataset layout supports either:
1) flat class folders
   <dataset_root>/H, <dataset_root>/S, <dataset_root>/U
2) split folders
   <dataset_root>/<train|valid|test>/<H|S|U>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


LABELS = ("H", "S", "U")
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class Sample:
    path: str
    class_index: int


class TrainLogger:
    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._fp = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass

    def line(self, message: str) -> None:
        print(message)
        self._fp.write(message + "\n")

    def header(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.line(f"[{stamp}] [RUN {self.run_id}] {message}")


def _find_samples(dataset_root: str) -> list[Sample]:
    split_train = os.path.join(dataset_root, "train")
    if os.path.isdir(split_train):
        return _find_split_samples(dataset_root, ("train", "valid", "test"))
    return _find_flat_samples(dataset_root)


def _find_flat_samples(dataset_root: str) -> list[Sample]:
    samples: list[Sample] = []
    for idx, label in enumerate(LABELS):
        class_dir = os.path.join(dataset_root, label)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for name in os.listdir(class_dir):
            path = os.path.join(class_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name.lower())[1] not in VALID_EXT:
                continue
            samples.append(Sample(path=path, class_index=idx))
    if not samples:
        raise RuntimeError(f"No training images found under {dataset_root}")
    return samples


def _find_split_samples(dataset_root: str, splits: Iterable[str]) -> list[Sample]:
    samples: list[Sample] = []
    for split in splits:
        split_root = os.path.join(dataset_root, split)
        if not os.path.isdir(split_root):
            continue
        for idx, label in enumerate(LABELS):
            class_dir = os.path.join(split_root, label)
            if not os.path.isdir(class_dir):
                continue
            for name in os.listdir(class_dir):
                path = os.path.join(class_dir, name)
                if not os.path.isfile(path):
                    continue
                if os.path.splitext(name.lower())[1] not in VALID_EXT:
                    continue
                samples.append(Sample(path=path, class_index=idx))
    if not samples:
        raise RuntimeError(f"No training images found under {dataset_root}")
    return samples


class SHUDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int, augment: bool = False) -> None:
        self.samples = samples
        self.input_size = input_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("L")
        arr = _preprocess_letter_gray(np.asarray(image, dtype=np.float32), self.input_size)
        if self.augment:
            # Mild geometric/photometric augmentation suitable for letter markers.
            if random.random() < 0.6:
                angle = random.uniform(-25.0, 25.0)
                im = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
                im = im.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                arr = np.asarray(im, dtype=np.float32) / 255.0
            if random.random() < 0.7:
                alpha = random.uniform(0.75, 1.25)
                beta = random.uniform(-0.10, 0.10)
                arr = np.clip(arr * alpha + beta, 0.0, 1.0)
            if random.random() < 0.5:
                noise = np.random.normal(0.0, 0.02, size=arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0.0, 1.0)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        label = torch.tensor(sample.class_index, dtype=torch.long)
        return tensor, label


class CompactNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc = nn.Linear(conv2_channels * pool_size * pool_size, num_classes)
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.pool_size = pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.head_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _preprocess_letter_gray(gray: np.ndarray, input_size: int) -> np.ndarray:
    # Robust contrast normalization.
    lo = float(np.percentile(gray, 5))
    hi = float(np.percentile(gray, 95))
    if hi - lo > 1e-6:
        gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    else:
        gray = np.clip(gray / 255.0, 0.0, 1.0)

    # Find foreground mask; choose bright or dark polarity based on compactness.
    bright_mask = gray > 0.62
    dark_mask = gray < 0.38
    b_count = int(np.count_nonzero(bright_mask))
    d_count = int(np.count_nonzero(dark_mask))
    total = gray.size

    use_bright = 0 < b_count < total * 0.75
    use_dark = 0 < d_count < total * 0.75
    if use_bright and use_dark:
        mask = bright_mask if b_count <= d_count else dark_mask
    elif use_bright:
        mask = bright_mask
    elif use_dark:
        mask = dark_mask
    else:
        mask = None

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

    # Resize to model input.
    im = Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")
    im = im.resize((input_size, input_size), Image.BILINEAR)
    out = np.asarray(im, dtype=np.float32) / 255.0
    return out


def _split(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_count = int(len(shuffled) * val_ratio)
    val = shuffled[:val_count]
    train = shuffled[val_count:]
    if not train:
        raise RuntimeError("Validation split too large; no train samples left")
    return train, val


def _find_split_samples_for(dataset_root: str, split: str) -> list[Sample]:
    samples: list[Sample] = []
    split_root = os.path.join(dataset_root, split)
    if not os.path.isdir(split_root):
        return samples
    for idx, label in enumerate(LABELS):
        class_dir = os.path.join(split_root, label)
        if not os.path.isdir(class_dir):
            continue
        for name in os.listdir(class_dir):
            path = os.path.join(class_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name.lower())[1] not in VALID_EXT:
                continue
            samples.append(Sample(path=path, class_index=idx))
    return samples


def _class_counts(samples: list[Sample]) -> list[int]:
    counts = [0 for _ in LABELS]
    for s in samples:
        counts[s.class_index] += 1
    return counts


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, list[float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    class_total = [0 for _ in LABELS]
    class_correct = [0 for _ in LABELS]
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="eval", leave=False)
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            pred = torch.argmax(logits, dim=1)
            total += y.size(0)
            correct += int((pred == y).sum().item())
            y_cpu = y.detach().cpu().numpy()
            p_cpu = pred.detach().cpu().numpy()
            for yi, pi in zip(y_cpu, p_cpu):
                class_total[int(yi)] += 1
                if int(yi) == int(pi):
                    class_correct[int(yi)] += 1
    if total == 0:
        return 0.0, 0.0, [0.0 for _ in LABELS]
    per_class_acc = []
    for c in range(len(LABELS)):
        if class_total[c] == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(class_correct[c] / class_total[c])
    return total_loss / total, correct / total, per_class_acc


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    logger: TrainLogger,
    class_weights: torch.Tensor | None = None,
) -> tuple[int, float]:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_epoch = 1
    best_val_acc = -1.0
    epoch_iter = range(1, epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="epochs")
    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        total = 0
        train_iter = train_loader
        if tqdm is not None:
            train_iter = tqdm(train_loader, desc=f"train e{epoch:03d}", leave=False)
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)
            if tqdm is not None:
                train_iter.set_postfix(loss=float(loss.item()))
        train_loss = total_loss / max(1, total)
        val_loss, val_acc, per_class_acc = _evaluate(model, val_loader, device)
        class_report = " ".join(f"{label}:{acc:.3f}" for label, acc in zip(LABELS, per_class_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        current_lr = optimizer.param_groups[0]["lr"]
        logger.line(
            f"[TRAIN][EPOCH {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} val_class_acc=[{class_report}] lr={current_lr:.6f}"
        )
        scheduler.step()
    return best_epoch, best_val_acc


def _export_weights(model: CompactNet, output_path: str, input_size: int) -> None:
    model_cpu = model.to("cpu")
    conv1_w = model_cpu.conv1.weight.detach().numpy().astype(np.float32)
    conv1_b = model_cpu.conv1.bias.detach().numpy().astype(np.float32)
    conv2_w = model_cpu.conv2.weight.detach().numpy().astype(np.float32)
    conv2_b = model_cpu.conv2.bias.detach().numpy().astype(np.float32)
    fc_w = model_cpu.fc.weight.detach().numpy().astype(np.float32)
    fc_b = model_cpu.fc.bias.detach().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        model_version=np.array("v2"),
        conv1_w=conv1_w,
        conv1_b=conv1_b,
        conv2_w=conv2_w,
        conv2_b=conv2_b,
        fc_w=fc_w,
        fc_b=fc_b,
        labels=np.array(LABELS),
        input_size=np.int32(input_size),
        pool_size=np.int32(model_cpu.pool_size),
    )
    print(f"saved_weights={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SHU compact CNN for Erebus controller")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing H/S/U folders")
    parser.add_argument(
        "--output",
        default="controllers/erebus_controller/models/cnn_shu_weights.npz",
        help="Output .npz path for runtime classifier",
    )
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conv1-channels", type=int, default=16, help="Conv1 channels")
    parser.add_argument("--conv2-channels", type=int, default=32, help="Conv2 channels")
    parser.add_argument(
        "--log-file",
        default="controllers/erebus_controller/logs/logs.txt",
        help="Path to append professional training logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = TrainLogger(args.log_file)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_split = _find_split_samples_for(args.dataset_root, "train")
    valid_split = _find_split_samples_for(args.dataset_root, "valid")
    if train_split and valid_split:
        train_samples, val_samples = train_split, valid_split
    else:
        all_samples = _find_samples(args.dataset_root)
        train_samples, val_samples = _split(all_samples, args.val_ratio, args.seed)

    train_ds = SHUDataset(train_samples, args.input_size, augment=True)
    val_ds = SHUDataset(val_samples, args.input_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_counts = _class_counts(train_samples)
    logger.header("TRAINING STARTED")
    logger.line(
        f"[TRAIN][CONFIG] dataset_root={args.dataset_root} output={args.output} "
        f"input_size={args.input_size} batch_size={args.batch_size} epochs={args.epochs} "
        f"lr={args.lr} seed={args.seed} conv1={args.conv1_channels} conv2={args.conv2_channels}"
    )
    logger.line(
        f"[TRAIN][DATA] device={device} train={len(train_ds)} val={len(val_ds)} "
        f"class_counts={dict(zip(LABELS, train_counts))}"
    )
    total = float(sum(train_counts))
    weights = []
    for c in train_counts:
        if c <= 0:
            weights.append(1.0)
        else:
            weights.append(total / (len(train_counts) * float(c)))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    model = CompactNet(
        num_classes=len(LABELS),
        conv1_channels=args.conv1_channels,
        conv2_channels=args.conv2_channels,
    ).to(device)
    best_epoch, best_val_acc = _train(
        model, train_loader, val_loader, args.epochs, args.lr, device, logger, class_weights=class_weights
    )
    _export_weights(model, args.output, args.input_size)
    logger.line(
        f"[TRAIN][SUMMARY] best_epoch={best_epoch:03d} best_val_acc={best_val_acc:.3f} "
        f"weights_path={args.output}"
    )
    logger.header("TRAINING FINISHED")
    logger.close()


if __name__ == "__main__":
    main()
