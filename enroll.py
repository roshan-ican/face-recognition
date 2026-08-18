"""Capture reference photos from the SAME camera you will recognise with."""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import cv2
import numpy as np
import face_recognition

from camera import no_camera_help, open_camera

KNOWN_FACES_DIR = "known_faces"

PROMPTS = [
    "look STRAIGHT at the lens",
    "turn your head slightly LEFT",
    "turn your head slightly RIGHT",
    "tilt your chin DOWN a little",
    "tilt your chin UP a little",
    "neutral again, but move to different lighting",
    "glasses off (or on, if they were off)",
    "a normal, relaxed expression",
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    name = sys.argv[1]
    cam_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len([f for f in os.listdir(person_dir) if f.startswith("cam-")])
    print(f"Enrolling '{name}' -> {person_dir}  ({existing} camera shots already)\n")

    cap, _ = open_camera(cam_index)
    if cap is None:
        no_camera_help(cam_index)
        sys.exit(1)

    print("SPACE = capture     Q = done")
    print("A window titled 'LOMAH enrol' should appear — check the taskbar\n")

    WINDOW = "LOMAH enrol"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW, 60, 60)
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.waitKey(1)

    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        small = np.ascontiguousarray(cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5))
        boxes = face_recognition.face_locations(small)

        preview = frame.copy()
        ok = len(boxes) == 1

        for (t, r, b, l) in boxes:
            t, r, b, l = t * 2, r * 2, b * 2, l * 2
            cv2.rectangle(preview, (l, t), (r, b),
                          (0, 255, 0) if ok else (0, 165, 255), 2)

        if not boxes:
            msg, col = "no face detected - do not capture", (0, 0, 255)
        elif len(boxes) > 1:
            msg, col = f"{len(boxes)} faces - only one person please", (0, 165, 255)
        else:
            msg, col = "ready - press SPACE", (0, 255, 0)

        cv2.putText(preview, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        hint = PROMPTS[(existing + saved) % len(PROMPTS)]
        cv2.putText(preview, f"shot {existing + saved + 1}: {hint}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            if not ok:
                print("  skipped — need exactly one clearly detected face")
                continue
            path = os.path.join(person_dir, f"cam-{existing + saved + 1:02d}.jpg")
            cv2.imwrite(path, frame)   # the ORIGINAL frame, not the annotated one
            saved += 1
            print(f"  saved {path}")

    cap.release()
    cv2.destroyAllWindows()

    total = existing + saved
    print(f"\n{saved} new shot(s). '{name}' now has {total} camera reference(s).")
    if total < 5:
        print("Aim for at least 5 — more angles and lighting means lower distances.")
    print("\nNow run:  python main.py", cam_index)


if __name__ == "__main__":
    main()
