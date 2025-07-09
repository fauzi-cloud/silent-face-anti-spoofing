import cv2
import os

def extract_frames(video_path, output_dir, video_name, frame_skip=5):
   
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir,
                f"{video_name}_frame_{saved:05d}.jpg"
            )
            cv2.imwrite(out_path, frame)
            saved += 1

        count += 1

    cap.release()

def process_dataset(input_root, output_root, label="fake", frame_skip=5):

    out_dir = os.path.join(output_root, label)
    os.makedirs(out_dir, exist_ok=True)

    for vid_file in os.listdir(input_root):
        if not vid_file.lower().endswith((".avi", ".mp4", ".mov")):
            continue

        vid_path = os.path.join(input_root, vid_file)
        video_name = os.path.splitext(vid_file)[0]

        extract_frames(vid_path, out_dir, video_name, frame_skip)
        print(f"✅ Processed {vid_file} → frames saved in {out_dir}")

if __name__ == "__main__":
    # Example usage:
    process_dataset(
        input_root="raw",
        output_root="data",
        label="fake",
        frame_skip=5
    )
