# Grok Image-to-Video Iteration Studio

This FastAPI app provides a human-in-the-loop image iteration workflow with an xAI/Grok-powered image-to-video handoff.

## Features

- Upload a reference image (R0)
- Automatic Style Card vision analysis
- 3-step micro-adjustment iteration with two candidates per step
- Human selection of winners to produce R1–R3
- Image-to-video generation (deferred) via xAI

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export XAI_API_KEY=your_key
export XAI_IMAGE_MODEL=grok-2-image
export XAI_VIDEO_MODEL=grok-2-video
export XAI_VISION_MODEL=grok-2-vision
export STORAGE_DIR=./storage
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

## Render Deployment

1. Create a new **Web Service** on Render and connect this repo.
2. Render uses `render.yaml` to configure the service.
3. Add the required environment variables in Render:
   - `XAI_API_KEY` (secret)
   - `XAI_IMAGE_MODEL`
   - `XAI_VIDEO_MODEL`
   - `XAI_VISION_MODEL`
   - `STORAGE_DIR`

Render will run:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## API Endpoints

- `POST /api/iterate` (multipart: file + prompt)
- `GET /api/iterate/{job_id}`
- `POST /api/iterate/{job_id}/choose`
- `POST /api/iterate/{job_id}/next`
- `POST /api/video-jobs`
- `GET /api/video-jobs/{job_id}`
- `GET /files/{job_id}/{filename}`
- `GET /health`
