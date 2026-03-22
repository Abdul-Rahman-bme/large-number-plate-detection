# 🚘 Large-Scale License Plate Detection with YOLO11

> Fine-tuning YOLO11s on a large Kaggle dataset to detect vehicle license plates in real-world conditions.

---

## 📌 Overview

This project trains a **YOLO11s** object detection model to localise vehicle license plates. It runs entirely on **Kaggle** (free T4 GPU) and covers the full ML pipeline:

- Dataset inspection & audit
- Custom YAML config for Kaggle paths
- Training with YOLO11s pretrained weights
- Evaluation on a held-out test split
- Prediction visualisation & error analysis

---

## 🗂️ Repository Structure

```
large-number-plate-detection/
├── large-number-plate-detection-v2.ipynb   # Main Kaggle notebook
├── README.md                               # This file
└── .gitignore                              # Ignores weights, caches, outputs
```

---

## 📦 Dataset

This notebook uses a **large license plate dataset** hosted on Kaggle. The dataset is expected to be mounted at:

```
/kaggle/input/<your-dataset-slug>/
    images/
        train/
        val/
        test/
    labels/
        train/
        val/
        test/
```

> 🔗 **[Add your Kaggle dataset link here]** — go to the dataset page on Kaggle and paste the URL.

The notebook auto-generates the correct `dataset.yaml` pointing to Kaggle paths — you do **not** need to edit any paths manually.

---

## 🚀 How to Run on Kaggle

### Step 1 — Fork or import the notebook

**Option A — Import from GitHub (recommended):**

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook → File → Import Notebook**
3. Paste the GitHub URL of this repo or the raw notebook URL:
   ```
   https://raw.githubusercontent.com/<your-username>/large-number-plate-detection/main/large-number-plate-detection-v2.ipynb
   ```

**Option B — Upload manually:**

1. Download `large-number-plate-detection-v2.ipynb` from this repo
2. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
3. `File → Import Notebook` → upload the `.ipynb` file

---

### Step 2 — Attach the dataset

1. In the Kaggle notebook editor, click **Add Data** (right panel)
2. Search for the license plate dataset and click **Add**
3. Update `DATASET_ROOT` in Cell 1 to match the mounted path:
   ```python
   DATASET_ROOT = "/kaggle/input/<your-dataset-slug>"
   ```

---

### Step 3 — Enable GPU

1. In the Kaggle notebook: **Settings → Accelerator → GPU T4 x2** (or GPU T4)
2. Training takes approximately **20–40 minutes** on a T4

---

### Step 4 — Run all cells

`Run All` — the notebook will:

1. ✅ Verify GPU availability
2. 📦 Install Ultralytics (`pip install ultralytics`)
3. 🖼️ Visually inspect a sample of training images
4. 🔍 Audit train / val / test splits for missing or mismatched labels
5. 📝 Write a `dataset.yaml` with correct Kaggle paths
6. 🧠 Load YOLO11s pretrained weights
7. 🏋️ Train for 50 epochs at 640px
8. 📊 Plot training curves (loss, mAP, precision, recall)
9. 🧪 Evaluate on the test split
10. 🔬 Visualise predictions and analyse errors

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | `yolo11s.pt` |
| Epochs | 50 |
| Image size | 640 × 640 |
| Classes | 1 (`license_plate`) |
| Device | CUDA (T4 GPU) |

---

## 📊 Output & Results

After training, results are saved to:

```
/kaggle/working/runs/plate_v1/
    weights/
        best.pt       ← use this for inference
        last.pt
    results.png
    confusion_matrix.png
    PR_curve.png
    F1_curve.png
```

To use the trained model for inference:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("your_image.jpg", conf=0.5)
results[0].show()
```

---

## 🛠️ Local Setup (optional)

If you want to run or experiment locally:

```bash
git clone https://github.com/<your-username>/large-number-plate-detection.git
cd large-number-plate-detection

pip install ultralytics jupyter
jupyter notebook large-number-plate-detection-v2.ipynb
```

> ⚠️ You'll need to update dataset paths to point to your local data directory.

---

## 📋 Requirements

- Python 3.8+
- `ultralytics` (YOLO11)
- `torch` + CUDA (for GPU training)
- `opencv-python`
- `matplotlib`

All dependencies are installed automatically inside the Kaggle notebook.

---

## 🗺️ Roadmap

- [ ] Add OCR stage (EasyOCR / PaddleOCR) to read detected plate text
- [ ] Export model to ONNX / TFLite for edge deployment
- [ ] Test on night-time and motion-blurred images
- [ ] Multi-class detection (plate + vehicle type)

---

## 📄 License

[MIT](LICENSE) — feel free to use, fork, and extend.

---

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the model framework
- Kaggle for free GPU compute
- Dataset authors (link your dataset source here)
