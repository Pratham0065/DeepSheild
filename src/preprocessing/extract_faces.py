import cv2
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_VIDEO_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "original_sequences"
    / "youtube"
    / "c23"
    / "videos"
)

FAKE_VIDEO_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "manipulated_sequences"
    / "Deepfakes"
    / "c23"
    / "videos"
)

REAL_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
FAKE_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "images" / "fake"

REAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ============================================================
# EXTRACT FACES FROM VIDEOS
# ============================================================

def extract_faces(video_dir, output_dir, label):
    video_files = list(video_dir.glob("*.mp4"))

    print(f"\n{label}: Found {len(video_files)} videos")

    total_faces = 0

    for video_path in video_files:

        print(f"Processing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))

        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Process every 10th frame
            if frame_number % 10 != 0:
                frame_number += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for face_number, (x, y, w, h) in enumerate(faces):

                face = frame[y:y+h, x:x+w]

                if face.size == 0:
                    continue

                face = cv2.resize(face, (224, 224))

                output_name = (
                    f"{video_path.stem}_"
                    f"frame_{frame_number}_"
                    f"face_{face_number}.jpg"
                )

                output_path = output_dir / output_name

                cv2.imwrite(str(output_path), face)

                total_faces += 1

            frame_number += 1

        cap.release()

    print(f"{label}: Extracted {total_faces} face images")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print(" DeepShield Face Extraction")
    print("========================================")

    extract_faces(
        REAL_VIDEO_DIR,
        REAL_OUTPUT_DIR,
        "REAL"
    )

    extract_faces(
        FAKE_VIDEO_DIR,
        FAKE_OUTPUT_DIR,
        "FAKE"
    )

    print("\nFace extraction completed!")