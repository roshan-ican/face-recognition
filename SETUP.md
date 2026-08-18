# Setup

## The error you hit, and why it lied to you

```
Please install `face_recognition_models` with this command before using `face_recognition`
```

That message does not mean the package is missing. It comes from
`face_recognition/api.py`:

```python
try:
    import face_recognition_models
except Exception:                 # <-- swallows the REAL error
    print("Please install `face_recognition_models` ...")
    quit()
```

`except Exception` catches *everything* and replaces it with one guess. The real
failure is hidden. To see it, import the module yourself:

```powershell
python -c "import face_recognition_models"
```

## The actual cause

You are on **Python 3.14** (`__pycache__/main.cpython-314.pyc`). This stack does
not run there:

| Package | Last released | Problem on 3.14 |
|---|---|---|
| `face_recognition_models` | 2017 | imports `pkg_resources`, which modern setuptools no longer provides |
| `face_recognition` | 2020 | unmaintained |
| `dlib` | — | no prebuilt wheels; compiles from source, needs Visual Studio Build Tools |

The `pip install git+...` also never finished — there is no `Successfully
installed` line in your output, only `Preparing metadata ... done`.

## Fix: a Python 3.12 virtual environment

Install Python 3.12 from python.org if you don't have it, then:

```powershell
cd C:\Users\amind\Documents\Learnings\face-recognition

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Verify before running anything else — this must print with no traceback:

```powershell
python -c "import dlib, face_recognition_models, face_recognition; print('ok')"
```

## Running it

Iriun shows up as an ordinary DirectShow camera, but usually **not** index 0 if
your laptop has a built-in webcam. Find it first:

```powershell
python main.py --list
```

Then run with that index (start the Iriun app on both phone and PC first):

```powershell
python main.py 1
```

Press `q` in the video window to quit.

## Notes on the code changes

- `np.ascontiguousarray(...)` around the RGB conversion. Without it dlib raises
  `Unsupported image type, must be 8bit gray or RGB image` — `cv2.cvtColor` can
  return a non-contiguous view, and dlib reads the raw buffer directly.
- `load_image_file(..., mode="RGB")`. Your `roshan-1.png` may carry an alpha
  channel, and dlib rejects 4-channel arrays.
- Detection now runs every 2nd frame and reuses the last result in between.
  Detection is the expensive part; drawing is free. Roughly doubles the frame rate.
- Camera index is a CLI argument instead of hardcoded `0`.
- Match test is `distance <= TOLERANCE` directly, instead of calling both
  `compare_faces` and `face_distance` — `compare_faces` is literally
  `face_distance(...) <= tolerance`, so the original did the same work twice.

Your previous file is kept as `main.py.bak`.
