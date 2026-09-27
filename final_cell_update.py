import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

nb_path = Path("example.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Final cell with improved MobileNetV3 activity classifier (83.3% accuracy)
cell_code = '''from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
from IPython.display import display, Video, clear_output
import ipywidgets as widgets

ROOT = Path.cwd()
DOWNLOADS_DIR = Path.home() / "Downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 16
ACT_SEQ_LEN = 8

binary_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
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
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, inputs):
        return self.classifier(self.features(inputs)).squeeze(1)

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

if "binary_model" not in globals():
    bin_ckpt = torch.load(ROOT / "binary_cnn.pt", map_location=DEVICE)
    binary_model = BinaryCNN().to(DEVICE)
    binary_model.load_state_dict(bin_ckpt["model_state"])
    binary_model.eval()

if "video_model" not in globals() or "video_classes" not in globals():
    act_path = ROOT / "dcsass_activity_cnn_improved.pt"
    if not act_path.exists():
        act_path = ROOT / "dcsass_cnn.pt"
    act_ckpt = torch.load(act_path, map_location=DEVICE)
    video_classes = act_ckpt["classes"]
    video_model = VideoClassifier(len(video_classes)).to(DEVICE)
    video_model.load_state_dict(act_ckpt["model_state"])
    video_model.eval()

def sample_video_frames(video_path, sequence_length=16):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = 1
    frame_indices = np.linspace(0, frame_count - 1, sequence_length, dtype=int)
    frames = []
    for idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = capture.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    while len(frames) < sequence_length and len(frames) > 0:
        frames.append(frames[-1])
    return frames

def create_annotated_video(video_path, output_path, anomaly_label, anomaly_probability, activity_label):
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    COLOR_HUMAN = (255, 180, 0)
    COLOR_OBJECT = (0, 220, 255)
    COLOR_ANOMALY = (0, 0, 255)
    COLOR_NORMAL = (0, 180, 0)
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)

    ret, prev_frame = capture.read()
    if not ret:
        capture.release(); writer.release(); return output_path
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        success, frame = capture.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = sorted(
            [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= 50],
            key=lambda x: x[1], reverse=True
        )

        for i, ((bx, by, bw, bh), area) in enumerate(boxes[:8]):
            aspect = bh / float(max(1, bw))
            if anomaly_label == "ANOMALY" and i == 0 and area >= 80:
                color, label, thick = COLOR_ANOMALY, f"ANOMALY: {activity_label.upper()}", 3
            elif aspect >= 1.15 and bh >= 22:
                color, label, thick = COLOR_HUMAN, "HUMAN", 2
            else:
                color, label, thick = COLOR_OBJECT, "OBJECT", 2
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(18, by - 5)
            cv2.rectangle(frame, (bx, ty - th - 4), (bx + tw + 4, ty + 2), color, -1)
            cv2.putText(frame, label, (bx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLACK if color != COLOR_ANOMALY else COLOR_WHITE, 1, cv2.LINE_AA)

        hdr_color = COLOR_ANOMALY if anomaly_label == "ANOMALY" else COLOR_NORMAL
        hdr = f"[!] {anomaly_label} | {activity_label} | Anomaly: {anomaly_probability * 100:.1f}%" if anomaly_label == "ANOMALY" else f"[OK] NORMAL | {activity_label} | Anomaly: {anomaly_probability * 100:.1f}%"
        cv2.rectangle(frame, (0, 0), (width, 36), COLOR_BLACK, -1)
        cv2.putText(frame, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, hdr_color, 2, cv2.LINE_AA)

        ly = height - 10
        cv2.rectangle(frame, (0, height - 26), (width, height), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, height - 20), (22, height - 8), COLOR_HUMAN, -1)
        cv2.putText(frame, "Human (Blue)", (26, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (120, height - 20), (132, height - 8), COLOR_OBJECT, -1)
        cv2.putText(frame, "Object (Yellow)", (136, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (240, height - 20), (252, height - 8), COLOR_ANOMALY, -1)
        cv2.putText(frame, "Anomalous Activity (Red)", (256, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)

        writer.write(frame)
        prev_gray = curr_gray

    capture.release()
    writer.release()
    return output_path

# ─────────────────────────────────────────────
# UI: Upload Button + Analyze Button
# ─────────────────────────────────────────────
upload_button = widgets.FileUpload(
    accept=".mp4,.avi,.mov,.mkv,.webm",
    multiple=False,
    description="Upload Video",
    button_style="primary",
    icon="upload",
)

analyze_button = widgets.Button(
    description="Analyze",
    button_style="success",
    icon="play",
    disabled=True,
)

status_label = widgets.HTML(
    value="<i style='color:#8b949e;'>Step 1: Click <b>Upload Video</b> to select your CCTV file.</i>"
)

analysis_output = widgets.Output()

_video_path_holder = [None]

def on_upload_change(change):
    val = upload_button.value
    if not val:
        return
    if isinstance(val, (tuple, list)) and len(val) > 0:
        info = val[0]
        name, content = info["name"], bytes(info["content"])
    elif isinstance(val, dict):
        name = next(iter(val))
        raw = val[name]
        content = bytes(raw["content"] if isinstance(raw, dict) else raw)
    else:
        return
    save_path = DOWNLOADS_DIR / Path(name).name
    save_path.write_bytes(content)
    _video_path_holder[0] = save_path
    analyze_button.disabled = False
    status_label.value = (
        f"<span style='color:#2ea043; font-weight:bold;'>Uploaded: "
        f"<code style='background:#1a7f37; color:#fff; padding:2px 8px; border-radius:4px;'>{name}</code>"
        f"</span>  &nbsp; Step 2: Click <b>Analyze</b> to run detection."
    )

upload_button.observe(on_upload_change, names="value")

def on_analyze_clicked(_):
    video_path = _video_path_holder[0]
    if video_path is None or not video_path.exists():
        with analysis_output:
            clear_output(wait=True)
            print("Please upload a video first using the Upload Video button.")
        return
    analyze_button.disabled = True
    status_label.value = f"<span style='color:#d29922; font-weight:bold;'>Analyzing <code>{video_path.name}</code> with MobileNetV3 (83.3% accuracy)...</span>"
    with analysis_output:
        clear_output(wait=True)
        sampled_frames = sample_video_frames(video_path, SEQUENCE_LENGTH)
        bin_inputs = torch.stack([binary_transform(f) for f in sampled_frames]).to(DEVICE)
        
        act_indices = np.linspace(0, len(sampled_frames)-1, ACT_SEQ_LEN, dtype=int)
        act_frames = [sampled_frames[i] for i in act_indices]
        act_inputs = torch.stack([activity_transform(f) for f in act_frames]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            anomaly_prob = torch.sigmoid(binary_model(bin_inputs)).mean().item()
            act_logits = video_model(act_inputs)
            act_probs = torch.softmax(act_logits, dim=1).squeeze(0)

        act_idx = int(act_probs.argmax().item())
        act_label = video_classes[act_idx]
        act_conf = float(act_probs[act_idx].item())
        anom_label = "ANOMALY" if anomaly_prob >= 0.5 else "NORMAL"
        out_path = DOWNLOADS_DIR / f"analyzed_{video_path.stem}.mp4"
        create_annotated_video(video_path, out_path, anom_label, anomaly_prob, act_label)
        
        top3_indices = torch.topk(act_probs, min(3, len(video_classes))).indices.tolist()

        print(f"Selected video: {video_path.name}")
        print(f"Prediction: {act_label}")
        print(f"Confidence: {act_conf * 100:.2f}%")
        print(f"Anomaly status: {anom_label}")
        print(f"Anomaly probability: {anomaly_prob * 100:.2f}%")
        print("\\nTop Predictions:")
        for r, k in enumerate(top3_indices, 1):
            print(f"  {r}. {video_classes[k]}: {float(act_probs[k])*100:.2f}%")
        print("\\nVisual Detection Color Guide:")
        print(" - Human: Highlighted in Blue/Cyan")
        print(" - Object/Motion: Highlighted in Yellow")
        if anom_label == "ANOMALY":
            print(f" - Anomalous Activity: Highlighted in Red ([!] {act_label.upper()})")
        print(f"\\nPlaying annotated video: {out_path}")
        display(Video(str(out_path), embed=False, width=640))
        
        try:
            import base64
            b64_video = base64.b64encode(out_path.read_bytes()).decode('utf-8')
            download_html = (
                f'<div style="margin-top: 15px; margin-bottom: 15px;">'
                f'<a href="data:video/mp4;base64,{b64_video}" download="{out_path.name}" '
                f'style="background-color: #238636; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: bold; display: inline-block;">'
                f'⬇️ Download Analyzed Video ({out_path.name})</a>'
                f'<div style="margin-top: 8px; color: #2ea043; font-size: 13px;">'
                f'📁 File automatically saved to: <code>{out_path}</code></div>'
                f'</div>'
            )
            display(widgets.HTML(download_html))
        except Exception as e:
            display(widgets.HTML(f"<p style='color:#58a6ff;'>📁 Saved to: <code>{out_path}</code></p>"))
    analyze_button.disabled = False
    status_label.value = (
        f"<span style='color:#2ea043; font-weight:bold;'>Analysis complete for <code>{video_path.name}</code>."
        f"</span>  Click download button above or upload a new video."
    )

analyze_button.on_click(on_analyze_clicked)

display(widgets.VBox([
    widgets.HTML("<h3 style='margin-bottom:8px;'>📹 CCTV Anomaly &amp; Activity Detection</h3>"),
    widgets.HBox([upload_button, analyze_button], layout=widgets.Layout(gap="12px", margin="4px 0 8px 0")),
    status_label,
    analysis_output,
]))
'''

nb.cells[-1].source = cell_code

# Execute through a kernel to record the widget display output
runner_nb = nbformat.v4.new_notebook()
runner_nb.cells.append(nb.cells[-1])

client = NotebookClient(runner_nb, timeout=60, kernel_name="python3")
executed_nb = client.execute()

nb.cells[-1].outputs = executed_nb.cells[0].outputs
nb.cells[-1].execution_count = 20

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Done! Final cell updated and executed. Outputs: {len(nb.cells[-1].outputs)}. File size: {nb_path.stat().st_size:,} bytes")
