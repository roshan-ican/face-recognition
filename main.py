"""Live face recognition against a folder of known faces."""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import cv2
import face_recognition
import numpy as np

from camera import no_camera_help, open_camera, probe

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6      # dlib's published threshold. Lower = stricter.
DOWNSCALE = 0.5      # DETECT on a half-size frame (fast)...
UPSAMPLE = 1         # ...raise to 2 if faces get missed; ~4x slower per step
PROCESS_EVERY = 2    # run detection every Nth frame, reuse the result between

ENCODER = "large"

ENROL_JITTERS = 10


def list_cameras() -> None:
    """Iriun is not necessarily index 0 — a built-in webcam usually wins."""
    print("Probing cameras (this takes a few seconds)...\n")
    found = probe()
    if not found:
        no_camera_help("any")
        return
    for i, backend, w, h in found:
        print(f"  [{i}] via {backend:<6} {w}x{h}")
    print("\nRun again with the index you want, e.g.  python main.py 1")


def load_known_faces():
    encodings, names = [], []

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"No '{KNOWN_FACES_DIR}' folder here. Expected layout:")
        print(f"  {KNOWN_FACES_DIR}/<person name>/<one or more photos>")
        sys.exit(1)

    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_file in sorted(os.listdir(person_dir)):
            image_path = os.path.join(person_dir, image_file)
            try:
                image = face_recognition.load_image_file(image_path, mode="RGB")
                boxes = face_recognition.face_locations(image)
                found = face_recognition.face_encodings(
                    image, boxes, num_jitters=ENROL_JITTERS, model=ENCODER
                )

                if not found:
                    print(f"  no face found in {image_path} — skipped")
                    continue

                encodings.append(found[0])
                names.append(person_name)
                print(f"  loaded {image_path} -> {person_name}")
            except Exception as e:
                print(f"  error loading {image_path}: {e}")

    return encodings, names


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] in ("--list", "-l"):
        list_cameras()
        return

    cam_index = int(args[0]) if args and args[0].isdigit() else 0

    print("Loading known faces...")
    known_encodings, known_names = load_known_faces()
    print(
        f"Loaded {len(known_encodings)} encoding(s) "
        f"for {len(set(known_names))} person/people\n"
    )

    if not known_encodings:
        print("Nothing to match against — every face will read as Unknown.")

    cap, _backend = open_camera(cam_index)
    if cap is None:
        no_camera_help(cam_index)
        sys.exit(1)

    # Fail loudly at startup rather than 300 frames into a blank feed.
    ok, first = cap.read()
    if ok and float(first.std()) < 8:
        print("\n  WARNING: the camera is streaming a FLAT image "
              f"(variation {first.std():.1f}) — there is no real video here.")
        print("  For Iriun that means the Windows client app is closed, or the")
        print("  phone disconnected. The virtual camera device stays registered")
        print("  either way, so Windows and OpenCV both report it as working.")
        print("  Open the Iriun app on both ends and confirm live video in its")
        print("  own preview window before running this.\n")

    print("Press 'q' to quit, 'd' toggles distance, 's' saves the frame.")
    print("A window titled 'LOMAH face recognition' should appear — it may")
    print("open BEHIND this terminal, so check your taskbar.\n")

    WINDOW = "LOMAH face recognition"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW, 60, 60)
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.waitKey(1)

    frame_no = 0
    show_distance = True
    cached = []  # last computed [(box, name, distance)], reused between frames
    scale = round(1 / DOWNSCALE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — camera dropped out.")
            print("If you Ctrl+C'd a previous run, the device may still be")
            print("claimed. Close the Iriun window, reopen it, and retry.")
            break

        if frame_no % PROCESS_EVERY == 0:
            rgb_full = np.ascontiguousarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            small = cv2.resize(rgb_full, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            rgb_small = np.ascontiguousarray(small)

            small_boxes = face_recognition.face_locations(
                rgb_small, number_of_times_to_upsample=UPSAMPLE
            )
            boxes = [
                (t * scale, r * scale, b * scale, l * scale)
                for (t, r, b, l) in small_boxes
            ]
            encodings = face_recognition.face_encodings(
                rgb_full, boxes, model=ENCODER
            )

            cached = []
            for box, encoding in zip(boxes, encodings):
                name, distance = "Unknown", None
                if known_encodings:
                    distances = face_recognition.face_distance(
                        known_encodings, encoding
                    )
                    best = int(np.argmin(distances))
                    distance = float(distances[best])
                    if distance <= TOLERANCE:
                        name = known_names[best]
                cached.append((box, name, distance))

        for (top, right, bottom, left), name, distance in cached:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = name
            if show_distance and distance is not None:
                label = f"{name}  d={distance:.3f}"
            cv2.putText(
                frame, label, (left, max(top - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        cv2.putText(
            frame, f"tolerance {TOLERANCE}", (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow(WINDOW, frame)

        if frame_no % 30 == 0:
            if cached:
                summary = ", ".join(
                    f"{n} d={d:.3f}" if d is not None else n
                    for _, n, d in cached
                )
            else:
                brightness = float(frame.mean())
                variation = float(frame.std())
                if variation < 8:
                    summary = (f"BLANK FEED (brightness {brightness:.0f}, "
                               f"variation {variation:.1f}) - the camera is "
                               f"streaming a flat image, not video")
                else:
                    summary = (f"no face (brightness {brightness:.0f}, "
                               f"variation {variation:.1f} - picture looks real)")
            print(f"  frame {frame_no:5d}  {summary}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nquit (q)")
            break
        if key == ord("d"):
            show_distance = not show_distance
        if key == ord("s"):
            cv2.imwrite("live_frame.jpg", frame)
            print("  saved live_frame.jpg - open it to see what the camera sees")

        if frame_no > 10 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            print("\nwindow closed")
            break

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\ninterrupted — use 'q' in the video window next time, it")
        print("releases the camera cleanly")
