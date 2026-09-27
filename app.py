import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
from pathlib import Path
import tempfile, os

st.set_page_config(page_title="CCTV Anomaly & Activity Detection", page_icon="📹", layout="centered")

st.markdown("""
<style>
body { background: #0d1117; }
.stApp { background: #0d1117; color: #e6edf3; }
h1 { color: #58a6ff; }
.stButton > button {
    background: #238636; color: white; border: none;
    border-radius: 8px; padding: 10px 28px; font-size: 16px;
    font-weight: bold; cursor: pointer;
}
.stButton > button:hover { background: #2ea043; }
.stFileUploader { border: 2px dashed #30363d; border-radius: 10px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 16
ACT_SEQ_LEN = 8

binary_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

activity_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x): return self.classifier(self.features(x)).squeeze(1)

class VideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.features = base.features
        self.pool = base.avgpool
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(576, 256), nn.Hardswish(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        B, T, C, H, W = x.shape
        x = self.features(x.view(B * T, C, H, W))
        x = self.flatten(self.pool(x))
        x = x.view(B, T, -1).mean(1)
        return self.classifier(x)

@st.cache_resource
def load_models():
    bin_ckpt = torch.load(ROOT / "binary_cnn.pt", map_location=DEVICE)
    binary_model = BinaryCNN().to(DEVICE)
    binary_model.load_state_dict(bin_ckpt["model_state"])
    binary_model.eval()

    act_path = ROOT / "dcsass_activity_cnn_improved.pt"
    if not act_path.exists():
        act_path = ROOT / "dcsass_cnn.pt"
    act_ckpt = torch.load(act_path, map_location=DEVICE)
    classes = act_ckpt["classes"]
    video_model = VideoClassifier(len(classes)).to(DEVICE)
    video_model.load_state_dict(act_ckpt["model_state"])
    video_model.eval()
    val_acc = act_ckpt.get("val_acc", 83.3)
    return binary_model, video_model, classes, val_acc

def sample_frames(video_path, n=16):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1
    idxs = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    while len(frames) < n and len(frames) > 0:
        frames.append(frames[-1])
    return frames

def annotate_video(video_path, out_path, anom_label, anom_prob, act_label):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    BLUE=(255,180,0); YELLOW=(0,220,255); RED=(0,0,255); GREEN=(0,180,0)
    ret, prev = cap.read()
    if not ret:
        cap.release(); writer.release(); return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thr = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        thr = cv2.dilate(thr, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted([(cv2.boundingRect(c), cv2.contourArea(c)) for c in cnts if cv2.contourArea(c)>=50], key=lambda x: x[1], reverse=True)
        for i, ((bx,by,bw,bh), area) in enumerate(boxes[:8]):
            aspect = bh / float(max(1,bw))
            if anom_label=="ANOMALY" and i==0 and area>=80:
                color,lbl,thick = RED, f"ANOMALY: {act_label.upper()}", 3
            elif aspect>=1.15 and bh>=22:
                color,lbl,thick = BLUE,"HUMAN",2
            else:
                color,lbl,thick = YELLOW,"OBJECT",2
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),color,thick)
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            ty = max(18,by-5)
            cv2.rectangle(frame,(bx,ty-th-4),(bx+tw+4,ty+2),color,-1)
            cv2.putText(frame,lbl,(bx+2,ty-2),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1,cv2.LINE_AA)
        hdr_c = RED if anom_label=="ANOMALY" else GREEN
        hdr = f"[!] {anom_label} | {act_label} | {anom_prob*100:.1f}%" if anom_label=="ANOMALY" else f"[OK] NORMAL | {act_label} | {anom_prob*100:.1f}%"
        cv2.rectangle(frame,(0,0),(w,36),(0,0,0),-1)
        cv2.putText(frame,hdr,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.58,hdr_c,2,cv2.LINE_AA)
        writer.write(frame)
        prev_gray = gray
    cap.release(); writer.release()

# ── UI ──────────────────────────────────────────────────────────
st.markdown("# 📹 CCTV Anomaly & Activity Detection")
st.markdown("Upload a CCTV video and click **Analyze** to detect anomalies and activities using the improved **MobileNetV3 model (83.3% accuracy)**.")

uploaded = st.file_uploader("📤 Upload Video", type=["mp4","avi","mov","mkv","webm"])

if uploaded:
    st.success(f"✅ Selected: **{uploaded.name}**")
    if st.button("▶ Analyze Video"):
        binary_model, video_model, classes, val_acc = load_models()
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / uploaded.name
            in_path.write_bytes(uploaded.read())
            out_path = Path(tmp) / f"analyzed_{in_path.stem}.mp4"

            with st.spinner("⏳ Analyzing video with improved activity model..."):
                frames = sample_frames(in_path, SEQUENCE_LENGTH)
                bin_inputs = torch.stack([binary_transform(f) for f in frames]).to(DEVICE)
                
                # Activity model takes 8 frames with activity_transform
                act_sample_indices = np.linspace(0, len(frames)-1, ACT_SEQ_LEN, dtype=int)
                act_frames = [frames[i] for i in act_sample_indices]
                act_inputs = torch.stack([activity_transform(f) for f in act_frames]).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    anom_prob = torch.sigmoid(binary_model(bin_inputs)).mean().item()
                    act_logits = video_model(act_inputs)
                    act_probs = torch.softmax(act_logits, dim=1).squeeze(0)

                act_idx = int(act_probs.argmax().item())
                act_label = classes[act_idx]
                act_conf = float(act_probs[act_idx].item())
                anom_label = "ANOMALY" if anom_prob >= 0.5 else "NORMAL"
                
                # Top 3 predictions for full visibility
                top3_indices = torch.topk(act_probs, min(3, len(classes))).indices.tolist()
                
                annotate_video(in_path, out_path, anom_label, anom_prob, act_label)

            # Results
            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("🎯 Detected Activity", act_label)
            col1.metric("📊 Activity Confidence", f"{act_conf*100:.1f}%")
            col2.metric("🚨 Anomaly Status", anom_label)
            col2.metric("⚠️ Anomaly Probability", f"{anom_prob*100:.1f}%")

            with st.expander("📈 Top Class Probabilities"):
                for rank, idx in enumerate(top3_indices, 1):
                    c_name = classes[idx]
                    c_prob = float(act_probs[idx].item())
                    st.write(f"**{rank}. {c_name}**: {c_prob*100:.1f}%")

            st.markdown("**Color Legend:**")
            st.markdown("🔵 Blue = Human &nbsp;|&nbsp; 🟡 Yellow = Object &nbsp;|&nbsp; 🔴 Red = Anomalous Activity")

            st.markdown("### 🎬 Annotated Video")
            video_bytes = out_path.read_bytes()
            st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Analyzed Video",
                data=video_bytes,
                file_name=f"analyzed_{in_path.stem}.mp4",
                mime="video/mp4",
                use_container_width=True
            )
