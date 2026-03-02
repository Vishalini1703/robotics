# CNN Model Tasks

This folder contains only CNN-related assets:
- CNN training/inference code
- CNN model weights
- SHU dataset splits (`train`, `valid`, `test`)

## Folder Structure

- `cnn_code/`
  - `cnn_classifier.py`
  - `train_cnn_shu.py`
- `cnn_models/`
  - `cnn_shu_weights.npz`
  - `cnn_shu_weights_v2_50e.npz`
- `cnn_dataset/`
  - `train/`
  - `valid/`
  - `test/`

## Requirements

Install Python packages:

```bash
pip install numpy pillow torch tqdm
```

`tqdm` is optional (training still works without it).

## Train a New CNN Model

Run from project root:

```bash
python model_related_tasks/cnn_code/train_cnn_shu.py --dataset-root model_related_tasks/cnn_dataset --output model_related_tasks/cnn_models/cnn_shu_weights_custom.npz --epochs 50 --batch-size 64 --input-size 32
```

Notes:
- `--dataset-root` supports split layout (`train/valid/test/<H|S|U>`) which is already used here.
- Output is an `.npz` model file compatible with `TinyCNNClassifier` in `cnn_classifier.py`.

## Use Existing Weights

Inside your Python code:

```python
import sys
sys.path.append("model_related_tasks/cnn_code")
from cnn_classifier import TinyCNNClassifier

clf = TinyCNNClassifier(weights_path="model_related_tasks/cnn_models/cnn_shu_weights_v2_50e.npz")
```
