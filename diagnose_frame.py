"""Why did no face get detected in this frame?"""

import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import cv2
import numpy as np
import face_recognition

path = sys.argv[1] if len(sys.argv) > 1 else "snapshot.jpg"

bgr = cv2.imread(path)
if bgr is None:
    print(f"Could not read {path}")
    sys.exit(1)

h, w = bgr.shape[:2]
rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
print(f"{path}: {w}x{h}, dtype={rgb.dtype}, contiguous={rgb.flags['C_CONTIGUOUS']}\n")


def attempt(label, image, **kw):
    try:
        boxes = face_recognition.face_locations(image, **kw)
    except Exception as e:
        print(f"  {label:<40} ERROR {e}")
        return None
    if boxes:
        t, r, b, l = boxes[0]
        print(f"  {label:<40} {len(boxes)} FOUND  box {r-l}x{b-t}px")
        return boxes
    print(f"  {label:<40} -")
    return None


print("HOG detector, varying upsample (each level doubles the image,")
print("finding smaller faces at roughly 4x the cost):")
winner = None
for n in (0, 1, 2):
    if attempt(f"hog, upsample={n}", rgb, number_of_times_to_upsample=n) and not winner:
        winner = f"number_of_times_to_upsample={n}"

print("\nRotations (a phone held sideways gives a 90-degree face, which the")
print("frontal detector cannot see at all):")
for name, code in (("90 CW", cv2.ROTATE_90_CLOCKWISE),
                   ("180", cv2.ROTATE_180),
                   ("90 CCW", cv2.ROTATE_90_COUNTERCLOCKWISE)):
    attempt(f"rotated {name}", np.ascontiguousarray(cv2.rotate(rgb, code)))

print("\nContrast normalisation (HOG reads gradients, so flat or backlit")
print("images give it very little to work with):")
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
l, a, b_ = cv2.split(lab)
l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
clahe_rgb = np.ascontiguousarray(
    cv2.cvtColor(cv2.cvtColor(cv2.merge((l2, a, b_)), cv2.COLOR_LAB2BGR),
                 cv2.COLOR_BGR2RGB))
attempt("CLAHE contrast boost", clahe_rgb)

print("\nUpscaled 2x (helps when the face is small in the frame):")
big = np.ascontiguousarray(cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
attempt("resized 2x, upsample=0", big, number_of_times_to_upsample=0)

print("\nCNN detector — far better on turned heads, glasses and odd lighting.")
print("Seconds per frame on CPU, so it is for stills and enrolment, not live video:")
cnn = attempt("cnn, upsample=0", rgb, model="cnn", number_of_times_to_upsample=0)

print("\n" + "=" * 60)
if winner:
    print(f"HOG works with {winner}")
    print("Set that in main.py and you are done.")
elif cnn:
    print("Only the CNN detector found the face.")
    print("That means the head is turned too far for HOG, or the glasses are")
    print("breaking the gradient pattern it looks for. Easiest fix by far:")
    print("look STRAIGHT at the phone lens and re-run --snap.")
else:
    print("Nothing found a face. Re-capture looking directly at the lens,")
    print("filling more of the frame, with light on your face rather than")
    print("behind you.")
