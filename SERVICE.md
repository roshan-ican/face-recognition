# Roshan face-recognition API

This is one local demo service. The caller sends `POST /recognize`; the API
opens Iriun camera 0, scans for Roshan for up to five seconds, releases the
camera, and returns the answer in the same HTTP response.

There is no roster upload, database, API key, job queue, or webhook callback.
Reference photos come from `known_faces/roshan/` and are encoded once when the
first recognition request arrives.

## Run

Start Iriun on the iPhone and Windows, then run:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

The equivalent direct command is:

```powershell
uvicorn face_service.main:app
```

Open `http://127.0.0.1:8000/docs` or verify the process with:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Trigger live recognition with:

```powershell
curl.exe -X POST http://127.0.0.1:8000/recognize
```

While recognition is running, the latest in-memory camera frame is available
as a JPEG from `GET /preview`. It returns `204` before the first frame. The
preview reuses recognition's camera frames, never opens a second camera, and is
not written to disk.

Example match response:

```json
{
  "approved": true,
  "status": "matched",
  "person": "roshan",
  "distance": 0.317,
  "cameraIndex": 0,
  "framesScanned": 2,
  "message": "roshan recognized"
}
```

The calling software should approve only when `approved` is `true`.

## Settings

Defaults are suitable for the current demo. To override them, copy
`.env.example` to `.env` and change the camera index, timeout, or tolerance.

To inspect available camera indexes:

```powershell
python test_setup.py --cams
```

## Test

```powershell
pytest
```
