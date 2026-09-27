import time, torch
from pathlib import Path

p = Path("dcsass_activity_cnn_improved.pt")
if p.exists():
    age = time.time() - p.stat().st_mtime
    print(f"Size: {p.stat().st_size:,} bytes")
    print(f"Last saved: {age:.0f} seconds ago")
    ckpt = torch.load(str(p), map_location="cpu")
    print(f"Val acc  : {ckpt.get('val_acc', 'N/A')}")
    print(f"Epoch    : {ckpt.get('epoch', 'N/A')}")
    print(f"Classes  : {ckpt.get('classes', [])}")
else:
    print("Model file not found yet")
