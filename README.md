# Compositional Scene Classification with EfficientNet

**IIT Hyderabad Deep Learning Hackathon 2026 (Kaggle)**

Binary image classification of synthetic 3D-rendered scenes, using an EfficientNet-B4 implemented and trained entirely from scratch (no pretrained weights, as required by the competition rules).

---

## Overview

Each image shows 3–5 simple geometric objects (spheres, cubes, cylinders) in varying colours, materials and sizes on a plain grey background. The rule that separates Class 0 from Class 1 was not disclosed. It appears to depend on which objects make up the scene, not on colour or appearance, which makes it a compositional reasoning task rather than a standard recognition task.

| Property | Details |
|---|---|
| Training images | 18,000 (9,000 per class) |
| Test images | 5,010 (labels withheld) |
| Format | PNG |
| Metric | Accuracy |

## Approach

- **Model:** EfficientNet-B4 written from scratch in PyTorch, using MBConv blocks, squeeze-and-excitation and stochastic depth.
- **Training:** Progressive resolution (128 → 192 → 224 px), AdamW with OneCycleLR, mixed precision, and Focal Loss with label smoothing.
- **Augmentation:** Flips, rotation, colour jitter, random grayscale and affine transforms to discourage colour shortcuts, plus Mixup/CutMix.
- **Inference:** A 3-model ensemble (different seeds) with 10-view test-time augmentation, giving 30 averaged predictions per image.
- **Pseudo-labelling:** Test images predicted with ≥ 0.95 confidence are added for a short fine-tune. In our final run no images met the threshold, so this step was skipped.

### Training stages

| Stage | Image size | Epochs | Batch size | Max LR | Augmentation |
|---|---|---|---|---|---|
| 1 | 128 | 8 | 128 | 1e-3 | Standard |
| 2 | 192 | 8 | 64 | 6e-4 | Aggressive |
| 3 | 224 | 25 | 48 | 3e-4 | Aggressive |

## Results

Kaggle test accuracy for every approach we tried:

| Model | Kaggle test accuracy |
|---|---|
| **EfficientNet-B4, 3-model ensemble + TTA (final)** | **75.54%** |
| EfficientNet-B2 | 75.41% |
| U-Net (adapted for classification) | 75.39% |
| EfficientNet-B0 | 75.34% |
| ResNet-18 | 75.31% |
| YOLOv5-style detector | 74.56% |
| Vanilla CNN | 66.58% |

### Limitations

- Validation accuracy reached about 99%, while test accuracy was about 75.5%. This large gap suggests the models learned cues that do not carry over to the test set, rather than the underlying rule.
- Most architectures scored within about 0.2% of each other, so the gains from the larger model and ensemble are small.
- cuDNN benchmarking is enabled for speed, so results can vary slightly between runs.

## Repository structure

```
.
├── final_submission.py   # Full pipeline: model, training, pseudo-labelling, TTA, submission
├── report.pdf            # Technical report
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Getting started

### Requirements

- Python 3.10+
- PyTorch 2.x and torchvision
- numpy, pandas, Pillow, scikit-learn

```bash
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. We trained on a single NVIDIA T4 (Kaggle free tier).

### Data layout

Download the competition data from Kaggle. The script expects:

```
<data_dir>/
├── train/
│   ├── 0/   # Class 0 images
│   └── 1/   # Class 1 images
└── test/    # Unlabelled test images
```

An extra nested level (`train/train/...`, `test/test/...`) is also handled automatically.

### Run

1. Set `data_dir` at the bottom of `final_submission.py` to your dataset path.
2. Update the output paths (`/kaggle/working/...`) if you are not running on Kaggle.
3. Run:

```bash
python final_submission.py
```

The script trains three models, runs ensemble inference with TTA and writes `predictions.csv` with columns `ID` and `TARGET`.

## References

1. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
2. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
3. Hu et al. (2018). *Squeeze-and-Excitation Networks.* CVPR.
4. Zhang et al. (2018). *mixup: Beyond Empirical Risk Minimization.* ICLR.
5. Yun et al. (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV.
6. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.* ICLR.
