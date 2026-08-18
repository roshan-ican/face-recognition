"""Diagnostics. Run these in order — each stage tests one thing."""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6


def stage_imports():
    print("=" * 60)
    print("1. IMPORTS")
    print("=" * 60)
    import numpy as np
    import cv2
    import dlib
    import face_recognition

    print(f"  python           {sys.version.split()[0]}")
    print(f"  numpy            {np.__version__}")
    print(f"  opencv           {cv2.__version__}")
    print(f"  dlib             {dlib.__version__}")
    print(f"  face_recognition {face_recognition.__version__}")
    print("  OK\n")


def stage_known_faces():
    import numpy as np
    import face_recognition

    print("=" * 60)
    print("2. KNOWN FACES")
    print("=" * 60)

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"  MISSING folder '{KNOWN_FACES_DIR}'")
        return [], []

    encodings, names = [], []
    for person in sorted(os.listdir(KNOWN_FACES_DIR)):
        pdir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(pdir):
            continue
        print(f"  {person}/")
        for fname in sorted(os.listdir(pdir)):
            path = os.path.join(pdir, fname)
            try:
                img = face_recognition.load_image_file(path, mode="RGB")
                h, w = img.shape[:2]
                boxes = face_recognition.face_locations(img)
                found = face_recognition.face_encodings(img, boxes)

                if not found:
                    print(f"    {fname:<28} {w}x{h}  NO FACE DETECTED")
                elif len(found) > 1:
                    print(f"    {fname:<28} {w}x{h}  {len(found)} faces — "
                          f"using the first; crop to one face for best results")
                    encodings.append(found[0]); names.append(person)
                else:
                    print(f"    {fname:<28} {w}x{h}  1 face  OK")
                    encodings.append(found[0]); names.append(person)
            except Exception as e:
                print(f"    {fname:<28} ERROR: {e}")

    print(f"\n  {len(encodings)} encoding(s), {len(set(names))} person/people")

    if encodings:
        d = face_recognition.face_distance([encodings[0]], encodings[0])[0]
        print(f"  self-distance check: {d:.4f} (expect 0.0000)")

    uniq = sorted(set(names))
    if len(uniq) > 1:
        print("\n  distance between people (must be > %.2f):" % TOLERANCE)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a = encodings[names.index(uniq[i])]
                b = encodings[names.index(uniq[j])]
                dist = face_recognition.face_distance([a], b)[0]
                flag = "OK" if dist > TOLERANCE else "TOO CLOSE"
                print(f"    {uniq[i]} vs {uniq[j]}: {dist:.3f}  {flag}")
    print()
    return encodings, names


def stage_cams():
    from camera import probe, no_camera_help

    print("=" * 60)
    print("3. CAMERAS  (tries MSMF, DSHOW and ANY on each index)")
    print("=" * 60)
    print("  probing, a few seconds...\n")

    found = probe()
    for i, backend, w, h in found:
        print(f"  [{i}] via {backend:<6} {w}x{h}")

    if not found:
        no_camera_help("any")
        return []

    indexes = sorted({i for i, _, _, _ in found})
    print(f"\n  usable indexes: {indexes}")
    print(f"  Try:  python test_setup.py --snap {indexes[-1]}")
    print()
    return indexes


def stage_snap(index):
    """The important test: one still frame, recognition run on it, saved to"""
    import cv2
    import numpy as np
    import face_recognition

    print("=" * 60)
    print(f"4. SNAPSHOT FROM CAMERA {index}")
    print("=" * 60)

    encodings, names = stage_known_faces()

    from camera import open_camera, no_camera_help

    cap, _backend = open_camera(index)
    if cap is None:
        no_camera_help(index)
        return

    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("  Opened, but could not read a frame.")
        return

    h, w = frame.shape[:2]
    print(f"  captured {w}x{h}")

    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes = face_recognition.face_locations(rgb)
    print(f"  faces detected: {len(boxes)}")

    if not boxes:
        print("\n  No face found. Usual causes, in order of likelihood:")
        print("    - too dark, or backlit (a window behind you)")
        print("    - too far away; fill more of the frame")
        print("    - phone held sideways, so the face is rotated 90 degrees")
        print("  Open snapshot.jpg and look at what the camera actually saw.")

    found = face_recognition.face_encodings(rgb, boxes)
    for (top, right, bottom, left), enc in zip(boxes, found):
        name, conf = "Unknown", None
        if encodings:
            dists = face_recognition.face_distance(encodings, enc)
            best = int(np.argmin(dists))
            print(f"\n  face at ({left},{top}) distances:")
            for n, d in sorted(zip(names, dists), key=lambda x: x[1]):
                print(f"    {n:<20} {d:.3f}")
            if dists[best] <= TOLERANCE:
                name, conf = names[best], 1 - dists[best]
                print(f"  -> MATCH: {name} (confidence {conf:.2f})")
            else:
                print(f"  -> no match; closest was {dists[best]:.3f}, "
                      f"tolerance is {TOLERANCE}")

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{name} ({conf:.2f})" if conf is not None else name
        cv2.putText(frame, label, (left, max(top - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite("snapshot.jpg", frame)
    print("\n  saved snapshot.jpg — open it and check the box is on your face")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--cams":
        stage_imports(); stage_cams()
    elif args and args[0] == "--snap":
        idx = int(args[1]) if len(args) > 1 else 0
        stage_imports(); stage_snap(idx)
    else:
        stage_imports(); stage_known_faces()
        print("Next:  python test_setup.py --cams")
