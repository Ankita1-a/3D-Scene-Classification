
import os
import gc
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# MEMORY CLEANUP
# ─────────────────────────────────────────────────────────────
def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU Memory: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────
# EFFICIENTNET-B4 FROM SCRATCH 
# ─────────────────────────────────────────────────────────────

def round_filters(filters, width_coeff, divisor=8):
    if width_coeff == 1.0:
        return filters
    filters *= width_coeff
    new_f = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_f < 0.9 * filters:
        new_f += divisor
    return int(new_f)


def round_repeats(repeats, depth_coeff):
    return int(math.ceil(depth_coeff * repeats))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_ch = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_ch, 1, bias=True),
            Swish(),
            nn.Conv2d(se_ch, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio,
                 se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.drop_connect_rate = drop_connect_rate
        mid_ch = in_ch * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
                       Swish()]
        pad = (kernel_size - 1) // 2
        layers += [nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                             padding=pad, groups=mid_ch, bias=False),
                   nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
                   Swish()]
        layers.append(SEBlock(mid_ch, se_ratio))
        layers += [nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)]
        self.block = nn.Sequential(*layers)

    def drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep = 1 - self.drop_connect_rate
        r = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        r = torch.floor(r + keep)
        return x / keep * r

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = self.drop_connect(out) + x
        return out


EFFICIENTNET_CONFIGS = {
    'b0': (1.0, 1.0, 224, 0.2),
    'b1': (1.0, 1.1, 240, 0.2),
    'b2': (1.1, 1.2, 260, 0.3),
    'b3': (1.2, 1.4, 300, 0.3),
    'b4': (1.4, 1.8, 380, 0.4),
}

MBCONV_SETTINGS = [
    [1, 16,  1, 1, 3],
    [6, 24,  2, 2, 3],
    [6, 40,  2, 2, 5],
    [6, 80,  3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]


class EfficientNet(nn.Module):
    def __init__(self, variant='b4', num_classes=2, dropout_rate=None):
        super().__init__()
        w, d, _, drop = EFFICIENTNET_CONFIGS[variant]
        if dropout_rate is None:
            dropout_rate = drop

        stem_ch = round_filters(32, w)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch, momentum=0.01, eps=1e-3),
            Swish()
        )

        total_blocks = sum(round_repeats(s[2], d) for s in MBCONV_SETTINGS)
        block_idx = 0
        blocks = []
        in_ch = stem_ch

        for expand, out, repeats, stride, ks in MBCONV_SETTINGS:
            out_ch = round_filters(out, w)
            n = round_repeats(repeats, d)
            for i in range(n):
                s = stride if i == 0 else 1
                dc = drop * block_idx / total_blocks
                blocks.append(MBConvBlock(in_ch, out_ch, ks, s, expand,
                                          drop_connect_rate=dc))
                in_ch = out_ch
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        head_ch = round_filters(1280, w)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch, momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Linear(head_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# FOCAL LOSS + LABEL SMOOTHING
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss — focuses on hard examples."""
    def __init__(self, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_cls    = pred.size(1)
        log_prob = F.log_softmax(pred, dim=1)
        prob     = log_prob.exp()

        # Label smoothing
        with torch.no_grad():
            smooth_target = torch.full_like(log_prob,
                                            self.smoothing / (n_cls - 1))
            smooth_target.scatter_(1, target.unsqueeze(1),
                                   1.0 - self.smoothing)

        # Focal weight
        p_t  = (prob * smooth_target).sum(dim=1)
        w    = (1 - p_t) ** self.gamma
        loss = -(w * (log_prob * smooth_target).sum(dim=1))
        return loss.mean()


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx] if self.labels is not None else -1
        return img, label


def build_transforms(img_size, mode='train', aggressive=False):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if mode == 'train':
        augs = [
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(20),
        ]
        if aggressive:
            augs += [
                T.ColorJitter(brightness=0.4, contrast=0.4,
                              saturation=0.4, hue=0.15),
                T.RandomGrayscale(p=0.1),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1),
                               scale=(0.85, 1.15), shear=10),
            ]
        augs += [
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.4, scale=(0.02, 0.25))
        ]
        return T.Compose(augs)
    else:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])


def load_train_data(data_dir):
    train_dir = Path(data_dir) / "train" / "train"
    if not train_dir.exists():
        train_dir = Path(data_dir) / "train"

    paths, labels = [], []
    for cls in ['0', '1']:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] Class dir not found: {cls_dir}")
            continue
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in sorted(cls_dir.glob(ext)):
                paths.append(str(img_path))
                labels.append(int(cls))

    print(f"[INFO] Train total: {len(paths)} images | "
          f"Class 0: {labels.count(0)} | Class 1: {labels.count(1)}")
    return paths, labels


def load_test_data(data_dir):
    test_dir = Path(data_dir) / "test" / "test"
    if not test_dir.exists():
        test_dir = Path(data_dir) / "test"

    paths, ids = [], []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img_path in sorted(test_dir.glob(ext)):
            paths.append(str(img_path))
            ids.append(img_path.name)

    print(f"[INFO] Test total: {len(paths)} images")
    return paths, ids


# ─────────────────────────────────────────────────────────────
# MIXUP / CUTMIX
# ─────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam      = np.random.beta(alpha, alpha)
    _, _, H, W = x.size()
    index    = torch.randperm(x.size(0), device=x.device)
    cut_h    = int(H * math.sqrt(1 - lam))
    cut_w    = int(W * math.sqrt(1 - lam))
    cx, cy   = random.randint(0, W), random.randint(0, H)
    x1, y1   = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2   = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    mixed    = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam      = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH 
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    scheduler=None, mixup_prob=0.5, scaler=None):
    model.train()
    total_loss, correct, total = 0., 0, 0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        use_mix = random.random() < mixup_prob
        if use_mix:
            if random.random() < 0.5:
                mixed, y_a, y_b, lam = mixup_data(imgs, labels)
            else:
                mixed, y_a, y_b, lam = cutmix_data(imgs, labels)
        else:
            mixed = imgs

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            preds = model(mixed)
            if use_mix:
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                loss = criterion(preds, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = preds.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast(enabled=torch.cuda.is_available()):
            _, predicted = model(imgs).max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


# ─────────────────────────────────────────────────────────────
# TTA
# ─────────────────────────────────────────────────────────────

def get_tta_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    hf   = T.RandomHorizontalFlip(p=1.0)
    vf   = T.RandomVerticalFlip(p=1.0)
    return [
        T.Compose([T.Resize((img_size+32, img_size+32)),
                   T.CenterCrop(img_size),
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size+32, img_size+32)),
                   T.CenterCrop(img_size), hf,
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size+32, img_size+32)),
                   T.RandomCrop(img_size),
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size+32, img_size+32)),
                   T.RandomCrop(img_size), hf,
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size, img_size)),
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size, img_size)), hf,
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size+64, img_size+64)),
                   T.CenterCrop(img_size),
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size+64, img_size+64)),
                   T.CenterCrop(img_size), vf,
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size+64, img_size+64)),
                   T.CenterCrop(img_size),
                   T.RandomRotation(10),
                   T.ToTensor(), T.Normalize(mean, std)]),
        T.Compose([T.Resize((img_size, img_size)),
                   vf, T.ToTensor(), T.Normalize(mean, std)]),
    ]


@torch.no_grad()
def predict_tta(models, test_paths, img_size, batch_size=32):
    all_tta_probs = []
    for tfm in get_tta_transforms(img_size):
        dataset = ImageDataset(test_paths, transform=tfm)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             num_workers=2, pin_memory=True)
        preds_per_model = [[] for _ in models]
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            for i, model in enumerate(models):
                model.eval()
                with autocast(enabled=torch.cuda.is_available()):
                    prob = F.softmax(model(imgs), dim=1)[:, 1]
                preds_per_model[i].append(prob.cpu().float().numpy())
        model_probs = [np.concatenate(p) for p in preds_per_model]
        all_tta_probs.append(np.mean(model_probs, axis=0))
    return np.mean(all_tta_probs, axis=0)


# ─────────────────────────────────────────────────────────────
# TRAINING STAGE
# ─────────────────────────────────────────────────────────────

def train_stage(model, train_paths, train_labels, val_paths, val_labels,
                img_size, epochs, batch_size, lr,
                stage_name="Stage", aggressive_aug=False,
                mixup_prob=0.5, checkpoint_path=None):

    free_memory()

    train_loader = DataLoader(
        ImageDataset(train_paths, train_labels,
                     build_transforms(img_size, 'train', aggressive_aug)),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        ImageDataset(val_paths, val_labels,
                     build_transforms(img_size, 'val')),
        batch_size=batch_size, num_workers=2, pin_memory=True
    )

    criterion = FocalLoss(gamma=2.0, smoothing=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.2,
        anneal_strategy='cos', div_factor=10,
        final_div_factor=100
    )

    # AMP scaler for mixed precision 
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_acc, best_state = 0.0, None

    print(f"\n{'='*60}")
    print(f" {stage_name} | img={img_size} | ep={epochs} | "
          f"bs={batch_size} | lr={lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scheduler=scheduler, mixup_prob=mixup_prob,
            scaler=scaler
        )
        val_acc = evaluate(model, val_loader)

        print(f"  Epoch {epoch:03d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.2f}% | "
              f"Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)

    print(f"  Best Val Acc ({stage_name}): {best_acc:.2f}%")
    if best_state:
        model.load_state_dict(best_state)

    free_memory()
    return best_acc


# ─────────────────────────────────────────────────────────────
# PSEUDO LABELING
# ─────────────────────────────────────────────────────────────

def pseudo_label_test(models, test_paths, img_size,
                      confidence_threshold=0.95, batch_size=32):
    probs = predict_tta(models, test_paths, img_size, batch_size)
    pseudo_paths, pseudo_labels = [], []
    for path, prob in zip(test_paths, probs):
        if prob >= confidence_threshold:
            pseudo_paths.append(path)
            pseudo_labels.append(1)
        elif prob <= (1 - confidence_threshold):
            pseudo_paths.append(path)
            pseudo_labels.append(0)
    print(f"[INFO] Pseudo-labeled {len(pseudo_paths)}/{len(test_paths)} "
          f"(threshold={confidence_threshold})")
    return pseudo_paths, pseudo_labels


# ─────────────────────────────────────────────────────────────
# MAIN: generate_predictions
# ─────────────────────────────────────────────────────────────

def generate_predictions(data_dir):

    set_seed(42)
    free_memory()

    train_paths, train_labels = load_train_data(data_dir)
    test_paths,  test_ids     = load_test_data(data_dir)

    from sklearn.model_selection import train_test_split
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=0.1, stratify=train_labels, random_state=42
    )
    print(f"[INFO] Train: {len(tr_paths)} | Val: {len(val_paths)}")

    # ── Stage config with AMP means we can use bigger batches ──
    # (img_size, epochs, batch_size, lr, aggressive_aug, mixup_prob)
    STAGES = [
        (128, 8,  128, 1e-3, False, 0.3),   
        (192, 8,   64, 6e-4, True,  0.4),  
        (224, 25,  48, 3e-4, True,  0.5),   
    ]

    NUM_MODELS = 3   
    all_models = []

    for model_idx in range(NUM_MODELS):
        set_seed(42 + model_idx * 13)
        free_memory()

        print(f"\n{'#'*60}")
        print(f"  TRAINING MODEL {model_idx + 1} / {NUM_MODELS}")
        print(f"{'#'*60}")

        model = EfficientNet(variant='b4', num_classes=2,
                             dropout_rate=0.4).to(DEVICE)

        for stage_idx, (img_size, epochs, bs, lr, agg, mp) in enumerate(STAGES):
            ckpt = f"/kaggle/working/model_{model_idx}_s{stage_idx}.pth"
            train_stage(
                model, tr_paths, tr_labels, val_paths, val_labels,
                img_size=img_size, epochs=epochs, batch_size=bs, lr=lr,
                stage_name=f"M{model_idx+1}-S{stage_idx+1}",
                aggressive_aug=agg, mixup_prob=mp,
                checkpoint_path=ckpt
            )

        all_models.append(model)

    # ── Pseudo labeling ────────────────────────────────────────
    print("\n[INFO] Generating pseudo labels...")
    pseudo_paths, pseudo_labels = pseudo_label_test(
        all_models, test_paths, img_size=224,
        confidence_threshold=0.95, batch_size=32
    )

    if len(pseudo_paths) > 200:
        combined_paths  = tr_paths  + pseudo_paths
        combined_labels = tr_labels + pseudo_labels
        for m_idx, model in enumerate(all_models):
            set_seed(999 + m_idx)
            free_memory()
            print(f"\n[INFO] Pseudo fine-tuning model {m_idx+1}...")
            train_stage(
                model, combined_paths, combined_labels,
                val_paths, val_labels,
                img_size=224, epochs=5, batch_size=48,
                lr=5e-5,
                stage_name=f"M{m_idx+1}-Pseudo",
                aggressive_aug=False, mixup_prob=0.2
            )
    else:
        print("[INFO] Too few pseudo labels — skipping.")

    # ── Final TTA + Save ───────────────────────────────────────
    print("\n[INFO] Running 10x TTA inference...")
    free_memory()
    final_probs = predict_tta(all_models, test_paths,
                              img_size=224, batch_size=32)
    final_preds = (final_probs >= 0.5).astype(int)

    submission = pd.DataFrame({"ID": test_ids, "TARGET": final_preds})
    out_path   = "/kaggle/working/predictions.csv"
    submission.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Saved: {out_path}")
    print(f"  Total : {len(final_preds)}")
    print(f"  Class 0: {int((final_preds==0).sum())} | "
          f"Class 1: {int((final_preds==1).sum())}")
    print(f"{'='*60}")
    print(submission.head(10).to_string(index=False))

    return submission

data_dir = "/kaggle/input/datasets/prathibhaavvariii/deeplearning" #write your own data directory
generate_predictions(data_dir)
