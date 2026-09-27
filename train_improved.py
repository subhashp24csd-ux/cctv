"""
Fast Activity CNN Training - Pre-extracts frames into RAM first, then trains.
CPU-friendly: ~50 videos per class, 8 frames each = manageable.
Uses MobileNetV3 pretrained backbone for high accuracy.
"""

import cv2, random, torch, numpy as np
from pathlib import Path
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from collections import Counter

# ── Config ──────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN       = 8          # frames per video (was 16, halved for speed)
BATCH_SIZE    = 16
EPOCHS        = 30
LR            = 3e-4
IMG_SIZE      = 96         # smaller than 112 for faster CPU training
MAX_PER_CLASS = 80         # 80 videos × 13 classes = 1040 total (fast!)
SAVE_PATH     = Path("dcsass_activity_cnn_improved.pt")
VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".mkv"}

print(f"Device: {DEVICE}")
print(f"Config: {MAX_PER_CLASS} videos/class, {SEQ_LEN} frames/video, {IMG_SIZE}px")

# ── Find dataset root ────────────────────────────────────────────
def find_dataset():
    for root in [Path("DCSASS Dataset") / "DCSASS Dataset",
                 Path("DCSASS Dataset"), Path("Train")]:
        if not root.exists():
            continue
        for cls_dir in root.iterdir():
            if not cls_dir.is_dir():
                continue
            vids = [f for f in cls_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
            if vids:
                print(f"Dataset found: {root}")
                return root
    raise RuntimeError("Dataset not found.")

DATASET_ROOT = find_dataset()

# ── Pre-extract frames into memory ───────────────────────────────
def extract_frames(video_path, n=SEQ_LEN, size=IMG_SIZE):
    cap = cv2.VideoCapture(str(video_path))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    last   = None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            f    = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size))
            last = f
        elif last is not None:
            f = last
        else:
            f = np.zeros((size, size, 3), dtype=np.uint8)
        frames.append(f)
    cap.release()
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]   # list of np arrays (H,W,3)

print("\nCollecting video paths...")
class_names, all_data, all_labels = [], [], []

for cls_dir in sorted(DATASET_ROOT.iterdir()):
    if not cls_dir.is_dir():
        continue
    vids = [f for f in cls_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    if not vids:
        continue
    random.shuffle(vids)
    vids = vids[:MAX_PER_CLASS]
    idx  = len(class_names)
    class_names.append(cls_dir.name)
    for v in vids:
        all_data.append(v)
        all_labels.append(idx)

print(f"Classes ({len(class_names)}): {class_names}")
print(f"Total videos: {len(all_data)}")
print(f"Per-class: {Counter(all_labels)}\n")

print("Pre-extracting frames into memory (this takes ~2-3 min)...")
frame_cache = []   # list of (frames_list, label)
for i, (vpath, label) in enumerate(zip(all_data, all_labels)):
    frames = extract_frames(vpath, SEQ_LEN, IMG_SIZE)
    frame_cache.append((frames, label))
    if (i + 1) % 50 == 0:
        print(f"  Extracted {i+1}/{len(all_data)} videos...")

print(f"Done! {len(frame_cache)} videos in memory.\n")

# ── Train/val split ──────────────────────────────────────────────
random.shuffle(frame_cache)
split      = int(0.85 * len(frame_cache))
tr_cache   = frame_cache[:split]
va_cache   = frame_cache[split:]

# ── Transforms ──────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Dataset from pre-extracted frames ───────────────────────────
class CachedVideoDataset(Dataset):
    def __init__(self, cache, transform):
        self.cache, self.tf = cache, transform
    def __len__(self): return len(self.cache)
    def __getitem__(self, i):
        frames, label = self.cache[i]
        tensors = torch.stack([self.tf(Image.fromarray(f)) for f in frames])
        return tensors, label

tr_ds = CachedVideoDataset(tr_cache, train_tf)
va_ds = CachedVideoDataset(va_cache, val_tf)

tr_labels = [x[1] for x in tr_cache]
counts     = Counter(tr_labels)
weights    = [1.0 / counts[l] for l in tr_labels]
sampler    = WeightedRandomSampler(weights, len(weights), replacement=True)

tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

# ── Model ────────────────────────────────────────────────────────
class VideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features   = base.features
        self.pool       = base.avgpool
        self.flatten    = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(576, 256), nn.Hardswish(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.features(x.view(B*T, C, H, W))
        x = self.flatten(self.pool(x))       # [B*T, 576]
        x = x.view(B, T, -1).mean(1)         # temporal pool → [B, 576]
        return self.classifier(x)

model = VideoClassifier(len(class_names)).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

# Class-weighted loss
total_tr = len(tr_labels)
wts = torch.tensor(
    [total_tr / (len(class_names) * counts.get(i, 1)) for i in range(len(class_names))],
    dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=wts)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ────────────────────────────────────────────────
best_val_acc = 0.0
print(f"{'Epoch':>5} {'Loss':>8} {'Train%':>8} {'Val%':>8}")
print("-" * 35)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = tr_correct = tr_total = 0
    for frames, labels in tr_loader:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(frames)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss    += loss.item() * labels.size(0)
        tr_correct += (out.argmax(1) == labels).sum().item()
        tr_total   += labels.size(0)
    scheduler.step()

    model.eval()
    va_correct = va_total = 0
    with torch.no_grad():
        for frames, labels in va_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            out = model(frames)
            va_correct += (out.argmax(1) == labels).sum().item()
            va_total   += labels.size(0)

    tr_acc = tr_correct / tr_total * 100
    va_acc = va_correct / va_total * 100
    flag   = ""
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save({"model_state": model.state_dict(),
                    "classes": class_names,
                    "val_acc": va_acc,
                    "epoch": epoch,
                    "img_size": IMG_SIZE,
                    "seq_len": SEQ_LEN}, SAVE_PATH)
        flag = "  ** SAVED"
    print(f"{epoch:>5} {tr_loss/tr_total:>8.4f} {tr_acc:>7.1f}% {va_acc:>7.1f}%{flag}")

print(f"\nBest val accuracy: {best_val_acc:.1f}%  ->  {SAVE_PATH}")
