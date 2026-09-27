"""
Run the final CCTV video analysis cell directly.
Can be executed from terminal:
    python run_final_cell.py
or with a custom video:
    python run_final_cell.py --video "path/to/video.mp4"
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = Path.home() / "Downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 16

valid_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
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

class SmallCNN(nn.Module):
    def __init__(self, number_of_classes):
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
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, number_of_classes)
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))

# Load checkpoints
bin_path = ROOT / "binary_cnn.pt"
bin_ckpt = torch.load(bin_path, map_location=DEVICE)
binary_model = BinaryCNN().to(DEVICE)
binary_model.load_state_dict(bin_ckpt["model_state"])
binary_model.eval()

act_path = ROOT / "dcsass_activity_cnn_improved.pt"
if not act_path.exists():
    act_path = ROOT / "dcsass_cnn.pt"
act_ckpt = torch.load(act_path, map_location=DEVICE)
video_classes = act_ckpt["classes"]
video_model = SmallCNN(len(video_classes)).to(DEVICE)
video_model.load_state_dict(act_ckpt["model_state"])
video_model.eval()

def sample_video_frames(video_path, sequence_length=16):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    frame_indices = np.linspace(0, frame_count - 1, sequence_length, dtype=int)
    frames = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if success:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames

def create_annotated_video(video_path, output_path, anomaly_label, anomaly_probability, activity_label):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create annotated video: {output_path}")

    # Color definitions
    COLOR_HUMAN = (255, 180, 0)      # Blue / Cyan
    COLOR_OBJECT = (0, 220, 255)     # Yellow
    COLOR_ANOMALY = (0, 0, 255)      # Red
    COLOR_NORMAL = (0, 180, 0)       # Green
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)

    ret, prev_frame = capture.read()
    if not ret:
        capture.release()
        writer.release()
        return output_path

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

        candidate_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= 50:
                bx, by, bw, bh = cv2.boundingRect(c)
                candidate_boxes.append((bx, by, bw, bh, area))

        candidate_boxes.sort(key=lambda b: b[4], reverse=True)

        for idx, (bx, by, bw, bh, area) in enumerate(candidate_boxes[:8]):
            aspect_ratio = bh / float(max(1, bw))
            
            # Anomalous Activity in Red
            if anomaly_label == "ANOMALY" and idx == 0 and area >= 80:
                box_color = COLOR_ANOMALY
                box_label = f"ANOMALY: {activity_label.upper()}"
                thickness = 3
            # Human in Blue
            elif aspect_ratio >= 1.15 and bh >= 22:
                box_color = COLOR_HUMAN
                box_label = "HUMAN"
                thickness = 2
            # Object / Motion in Yellow
            else:
                box_color = COLOR_OBJECT
                box_label = "OBJECT"
                thickness = 2

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, thickness)
            
            (text_w, text_h), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y = max(18, by - 5)
            cv2.rectangle(frame, (bx, tag_y - text_h - 4), (bx + text_w + 4, tag_y + 2), box_color, -1)
            cv2.putText(frame, box_label, (bx + 2, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLACK if box_color != COLOR_ANOMALY else COLOR_WHITE, 1, cv2.LINE_AA)

        # Top alert banner
        banner_color = COLOR_ANOMALY if anomaly_label == "ANOMALY" else COLOR_NORMAL
        header_text = f"[!] {anomaly_label} | {activity_label} | Anomaly: {anomaly_probability * 100:.1f}%" if anomaly_label == "ANOMALY" else f"[OK] NORMAL | {activity_label} | Anomaly: {anomaly_probability * 100:.1f}%"
        cv2.rectangle(frame, (0, 0), (width, 36), COLOR_BLACK, -1)
        cv2.putText(frame, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, banner_color, 2, cv2.LINE_AA)

        # Bottom visual legend
        legend_y = height - 10
        cv2.rectangle(frame, (0, height - 26), (width, height), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, height - 20), (22, height - 8), COLOR_HUMAN, -1)
        cv2.putText(frame, "Human (Blue)", (26, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (120, height - 20), (132, height - 8), COLOR_OBJECT, -1)
        cv2.putText(frame, "Object (Yellow)", (136, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (240, height - 20), (252, height - 8), COLOR_ANOMALY, -1)
        cv2.putText(frame, "Anomalous Activity (Red)", (256, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)

        writer.write(frame)
        prev_gray = curr_gray

    capture.release()
    writer.release()
    return output_path

def run_analysis(video_path):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    sampled_frames = sample_video_frames(video_path, SEQUENCE_LENGTH)
    model_inputs = torch.stack([valid_transform(frame) for frame in sampled_frames]).to(DEVICE)
    with torch.no_grad():
        anomaly_probability = torch.sigmoid(binary_model(model_inputs)).mean().item()
        activity_probabilities = torch.softmax(video_model(model_inputs), dim=1).mean(dim=0)
    activity_index = int(activity_probabilities.argmax().item())
    activity_label = video_classes[activity_index]
    activity_confidence = float(activity_probabilities[activity_index].item())
    anomaly_label = "ANOMALY" if anomaly_probability >= 0.5 else "NORMAL"

    annotated_path = create_annotated_video(
        video_path, DOWNLOADS_DIR / f"analyzed_{video_path.stem}.mp4", anomaly_label, anomaly_probability, activity_label
    )

    print(f"Selected video: {video_path}")
    print(f"Sampled frames: {len(sampled_frames)}")
    print(f"Prediction: {activity_label}")
    print(f"Confidence: {activity_confidence * 100:.2f}%")
    print(f"Anomaly status: {anomaly_label}")
    print(f"Anomaly probability: {anomaly_probability * 100:.2f}%")
    print("\nVisual Detection Color Guide:")
    print(" - Human: Highlighted in Blue/Cyan")
    print(" - Object/Motion: Highlighted in Yellow")
    if anomaly_label == "ANOMALY":
        print(f" - Anomalous Activity: Highlighted in Red ([!] {activity_label.upper()})")
    print(f"\nGenerated annotated video: {annotated_path}")
    return annotated_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a CCTV video for anomalies and activities")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()

    if args.video:
        target = Path(args.video)
    else:
        sample = DOWNLOADS_DIR / "Fighting002_x264_8.mp4"
        if sample.exists():
            target = sample
        else:
            candidates = list(DOWNLOADS_DIR.glob("*.mp4"))
            if not candidates:
                raise FileNotFoundError("No MP4 video found in Downloads")
            target = candidates[0]

    run_analysis(target)
