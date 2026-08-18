"""Opening a webcam on Windows, reliably."""

import cv2


def silence_logs():
    """Silence OpenCV's C++ logger."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        try:
            cv2.setLogLevel(0)
        except Exception:
            pass


BACKENDS = [
    (cv2.CAP_MSMF, "MSMF"),
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_ANY, "ANY"),
]


def safe_read(cap):
    """(ok, frame), never raising — cap.read() throws on a mismatched size."""
    try:
        ok, frame = cap.read()
        return bool(ok), frame
    except cv2.error:
        return False, None


def open_camera(index, width=None, height=None, quiet=False):
    """Return (capture, backend_name), or (None, None) if nothing works."""
    silence_logs()
    for flag, name in BACKENDS:
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = safe_read(cap)          # the only test that means anything
        if not ok:
            cap.release()
            continue

        if width and height:
            native = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if native != (width, height):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if not safe_read(cap)[0]:
                    cap.release()
                    cap = cv2.VideoCapture(index, flag)
                    if not (cap.isOpened() and safe_read(cap)[0]):
                        cap.release()
                        continue
                    if not quiet:
                        print(f"Camera {index} does not offer {width}x{height}"
                              f" — using its native {native[0]}x{native[1]}")

        if not quiet:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera {index} opened via {name} at {w}x{h}")
        return cap, name

    return None, None


def probe(max_index=5):
    """Every (index, backend, width, height) that actually yields a frame."""
    silence_logs()
    found = []
    for i in range(max_index):
        for flag, name in BACKENDS:
            cap = cv2.VideoCapture(i, flag)
            ok, frame = safe_read(cap) if cap.isOpened() else (False, None)
            if ok:
                h, w = frame.shape[:2]
                found.append((i, name, w, h))
            cap.release()
    return found


def no_camera_help(index):
    print(f"\nCould not read a frame from camera {index} on any backend.\n")
    print("Checklist, in the order that actually fixes it:")
    print("  1. Is the Iriun app open on BOTH the phone and Windows, showing")
    print("     live video in the Windows preview window? The virtual camera")
    print("     publishes no video format until it is genuinely streaming —")
    print("     that is exactly the 'pVih' error.")
    print("  2. Close anything else holding a camera: Teams, Zoom, Chrome tabs,")
    print("     the Windows Camera app. Only one process gets the device.")
    print("  3. Settings > Privacy & security > Camera:")
    print("     'Let desktop apps access your camera' must be On.")
    print("  4. Re-run  python test_setup.py --cams  to see the live list.")
