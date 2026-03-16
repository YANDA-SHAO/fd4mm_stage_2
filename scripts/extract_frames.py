import cv2
import os
import argparse

def extract(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name = os.path.join(out_dir, f"{frame_id:06d}.png")
        cv2.imwrite(name, frame)

        frame_id += 1

    cap.release()

    print("Total frames:", frame_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    extract(args.video, args.out)