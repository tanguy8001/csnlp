import os
import cv2
import subprocess
import webvtt

def download_video_and_caption(video_id, output_dir="videos/"):
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    vtt_path = os.path.join(output_dir, f"{video_id}.en.vtt")

    # Skip if both already exist
    if os.path.exists(video_path) and os.path.exists(vtt_path):
        return video_path, vtt_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--write-sub",
        "--sub-lang", "en",
        "--output", f"{output_dir}/%(id)s.%(ext)s",
        url
    ]
    subprocess.run(cmd, check=True)

    return video_path, vtt_path


def extract_caption_text(video_id, output_dir="videos/"):
    vtt_path = os.path.join(output_dir, f"{video_id}.en.vtt")
    
    if not os.path.exists(vtt_path):
        print(f"[WARN] Caption file not found: {vtt_path}")
        return None

    try:
        captions = list(webvtt.read(vtt_path))
        print(f"[INFO] Found {len(captions)} caption segments in {video_id}")
        for i, caption in enumerate(captions[:3]):
            print(f"  [{i}] {caption.start} --> {caption.end}: {caption.text}")
        return " ".join([c.text for c in captions])
    except Exception as e:
        print(f"[ERROR] Failed to read VTT for {video_id}: {e}")
        return None


def sample_frames(video_path, frame_rate=1, max_frames=256):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if int(i % round(fps / frame_rate)) == 0:
            frames.append(frame)
        i += 1
    cap.release()
    
    # Pad to max_frames
    while len(frames) < max_frames:
        frames.append(frames[-1])  # repeat last frame
    return frames[:max_frames] 