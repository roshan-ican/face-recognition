"""Are your reference photos actually giving the matcher anything new?"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import numpy as np
import face_recognition

KNOWN_FACES_DIR = "known_faces"
ENCODER = "large"


def encode_folder(person_dir):
    out = []
    for fname in sorted(os.listdir(person_dir)):
        path = os.path.join(person_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            img = face_recognition.load_image_file(path, mode="RGB")
        except Exception as e:
            print(f"  {fname:<18} unreadable: {e}")
            continue
        boxes = face_recognition.face_locations(img)
        encs = face_recognition.face_encodings(img, boxes, num_jitters=10,
                                               model=ENCODER)
        if not encs:
            print(f"  {fname:<18} NO FACE — this file is dead weight")
            continue
        out.append((fname, encs[0]))
    return out


people = {}
for person in sorted(os.listdir(KNOWN_FACES_DIR)):
    pdir = os.path.join(KNOWN_FACES_DIR, person)
    if os.path.isdir(pdir):
        print(f"\n{person}/")
        people[person] = encode_folder(pdir)

for person, items in people.items():
    if len(items) < 2:
        continue
    print(f"\n{'=' * 62}\n{person}: distance between your own photos\n{'=' * 62}")

    names = [n for n, _ in items]
    width = max(len(n) for n in names) + 1
    print(" " * width + " ".join(f"{n[:8]:>8}" for n in names))

    pairs = []
    for i, (ni, ei) in enumerate(items):
        row = []
        for j, (nj, ej) in enumerate(items):
            d = 0.0 if i == j else float(np.linalg.norm(ei - ej))
            row.append(f"{d:8.3f}")
            if i < j:
                pairs.append((d, ni, nj))
        print(f"{ni:<{width}}" + " ".join(row))

    if not pairs:
        continue

    ds = [d for d, _, _ in pairs]
    print(f"\n  spread: min {min(ds):.3f}   mean {np.mean(ds):.3f}   max {max(ds):.3f}")

    dupes = [(d, a, b) for d, a, b in pairs if d < 0.15]
    if dupes:
        print(f"\n  {len(dupes)} near-duplicate pair(s) — these add nothing:")
        for d, a, b in sorted(dupes)[:6]:
            print(f"    {a} vs {b}: {d:.3f}")

    if np.mean(ds) < 0.2:
        print("\n  VERDICT: your references are all basically the same shot.")
        print("  Re-run enroll.py and genuinely change something between each")
        print("  capture — turn your head, move to different light, glasses")
        print("  off. Right now 4 photos are doing the work of about 1.")
    elif np.mean(ds) < 0.45:
        print("\n  VERDICT: good coverage. This is what you want.")
    else:
        print("\n  VERDICT: very spread out. Check every photo is really you")
        print("  and that none is blurred or badly boxed.")

# Different people must be FAR apart, or they will be confused for each other.
if len(people) > 1:
    print(f"\n{'=' * 62}\nBetween different people (want > 0.6)\n{'=' * 62}")
    names = list(people)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = [e for _, e in people[names[i]]]
            b = [e for _, e in people[names[j]]]
            if not a or not b:
                continue
            worst = min(float(np.linalg.norm(x - y)) for x in a for y in b)
            flag = "OK" if worst > 0.6 else "TOO CLOSE — will be confused"
            print(f"  {names[i]} vs {names[j]}: closest pair {worst:.3f}  {flag}")
