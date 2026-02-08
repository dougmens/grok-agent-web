import base64
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from xai_sdk import Client


STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "./storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

XAI_API_KEY = os.environ.get("XAI_API_KEY")
XAI_IMAGE_MODEL = os.environ.get("XAI_IMAGE_MODEL", "grok-2-image")
XAI_VIDEO_MODEL = os.environ.get("XAI_VIDEO_MODEL", "grok-2-video")
XAI_VISION_MODEL = os.environ.get("XAI_VISION_MODEL", "grok-2-vision")

STYLE_PROMPT = (
    "You are a vision analyst. Return ONLY valid JSON. "
    "Create a Style Card with fields: scene, subject, wardrobe, palette, lighting, lens, mood, "
    "composition, constraints, safety. Respect constraints: adults only, non-explicit, no graphic nudity, "
    "no explicit sexual acts, cinematic/editorial/intimate aesthetic only."
)

CANDIDATE_PROMPT = (
    "You are a vision analyst. Return ONLY valid JSON with fields: "
    "description, framing, lighting, privacy, consistency, artifacts."
)

MICRO_DELTAS = {
    1: "Tighter framing with a closer camera. Preserve the same subject and mood.",
    2: "Darker, warmer lighting with a more private atmosphere. Preserve composition.",
    3: "Slightly reduced physical distance between subjects. Preserve intimacy and non-explicit tone.",
}

SAFETY_CONSTRAINTS = (
    "Adults only. Non-explicit. No graphic nudity. No explicit sexual acts. "
    "Cinematic/editorial/intimate aesthetic only. Do not violate safety constraints."
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@dataclass
class Candidate:
    id: str
    image_path: str
    description: Dict[str, Any]
    score: int


@dataclass
class IterateJob:
    id: str
    prompt: str
    created_at: str
    status: str = "processing"
    step: int = 0
    style_card: Optional[Dict[str, Any]] = None
    candidates: Dict[int, List[Candidate]] = field(default_factory=dict)
    chosen: Dict[int, str] = field(default_factory=dict)
    final_image_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class VideoJob:
    id: str
    status: str
    image_url: str
    prompt: Optional[str]
    duration: int
    aspect_ratio: str
    resolution: str
    request_id: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None


ITERATE_JOBS: Dict[str, IterateJob] = {}
VIDEO_JOBS: Dict[str, VideoJob] = {}
LOCK = threading.Lock()


class ChooseRequest(BaseModel):
    choice_id: str


class VideoJobRequest(BaseModel):
    image_url: str
    prompt: Optional[str] = None
    duration: int = 6
    aspect_ratio: str = "16:9"
    resolution: str = "720p"


def _client() -> Client:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is not set")
    return Client(api_key=XAI_API_KEY)


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _vision_json(data_uri: str, prompt: str) -> Dict[str, Any]:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is not set")
    payload = {
        "model": XAI_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {XAI_API_KEY}"},
        json=payload,
        timeout=120,
    )
    status_code = response.status_code
    raw_text = response.text
    if status_code != 200:
        raise Exception(f"HTTP {status_code}: {raw_text[:500]}")
    try:
        response_data = response.json()
    except json.JSONDecodeError as exc:
        raise Exception(f"Invalid JSON response from xAI: {raw_text[:500]}") from exc
    choices = response_data.get("choices")
    if not choices or not isinstance(choices, list):
        raise Exception(f"Missing choices in xAI response: {raw_text[:500]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not content or not isinstance(content, str) or not content.strip():
        raise Exception(f"Missing content in xAI response: {raw_text[:500]}")
    extracted = _extract_json_content(content)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as exc:
        snippet = content[:500]
        raise Exception(f"Failed to parse model JSON output: {snippet}") from exc


def _extract_json_content(content: str) -> str:
    fence_start = content.find("```")
    if fence_start != -1:
        fence_end = content.find("```", fence_start + 3)
        if fence_end != -1:
            fenced = content[fence_start + 3:fence_end].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            return fenced
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise Exception(f"Unable to locate JSON object in model output: {content[:500]}")
    return content[start:end + 1]


def _score_candidate(desc: Dict[str, Any], step: int) -> int:
    text = " ".join(str(v).lower() for v in desc.values())
    score = 50
    bonuses = {
        1: ["close", "tight", "framing"],
        2: ["warm", "dark", "moody", "private"],
        3: ["intimate", "close", "distance"],
    }
    penalties = ["blurry", "artifact", "distorted", "glitch", "noise"]
    for word in bonuses.get(step, []):
        if word in text:
            score += 10
    for word in penalties:
        if word in text:
            score -= 15
    score = max(0, min(100, score))
    return score


def _build_image_prompt(style_card: Dict[str, Any], user_prompt: str, step: int) -> str:
    style_json = json.dumps(style_card, ensure_ascii=False)
    micro = MICRO_DELTAS[step]
    parts = [
        "Style Card:",
        style_json,
        "User prompt:",
        user_prompt or "(none)",
        "Micro adjustment:",
        micro,
        "Safety:",
        SAFETY_CONSTRAINTS,
    ]
    return "\n".join(parts)


def _generate_candidates(job: IterateJob, step: int) -> None:
    job.status = "processing"
    job.step = step
    job_dir = STORAGE_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    reference_path = job_dir / f"R{step - 1}.png"
    reference_uri = _encode_image(reference_path)
    prompt = _build_image_prompt(job.style_card or {}, job.prompt, step)

    client = _client()
    candidates: List[Candidate] = []
    for idx in range(2):
        image = client.image.sample(
            model=XAI_IMAGE_MODEL,
            prompt=prompt,
            image=reference_uri,
            size="1024x1024",
        )
        image_bytes = base64.b64decode(image.b64_json)
        candidate_id = f"s{step}c{idx + 1}"
        image_path = job_dir / f"{candidate_id}.png"
        image_path.write_bytes(image_bytes)
        try:
            desc = _vision_json(_encode_image(image_path), CANDIDATE_PROMPT)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                desc = {
                    "description": "",
                    "framing": "",
                    "lighting": "",
                    "privacy": "",
                    "consistency": "",
                    "artifacts": "",
                }
            else:
                raise
        score = _score_candidate(desc, step)
        candidates.append(
            Candidate(id=candidate_id, image_path=str(image_path), description=desc, score=score)
        )
    job.candidates[step] = candidates
    job.status = "awaiting_choice"


def _start_iteration(job_id: str) -> None:
    job = ITERATE_JOBS[job_id]
    try:
        job.status = "processing"
        job.step = 0
        job_dir = STORAGE_DIR / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        r0_path = job_dir / "R0.png"
        try:
            job.style_card = _vision_json(_encode_image(r0_path), STYLE_PROMPT)
            job.step = 1
            job.status = "waiting_choice"
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                job.style_card = {
                    "people_count": None,
                    "clothing_style": "",
                    "setting": "",
                    "lighting": "",
                    "camera_framing": "",
                    "mood_keywords": [],
                }
                job.error = "Vision unavailable (403), proceeding with defaults"
                job.status = "waiting_choice"
                job.step = 1
            else:
                raise
        _generate_candidates(job, 1)
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)


def _start_video_job(job_id: str) -> None:
    job = VIDEO_JOBS[job_id]
    try:
        client = _client()
        response = client.video.start(
            model=XAI_VIDEO_MODEL,
            image_url=job.image_url,
            prompt=job.prompt or "",
            duration=job.duration,
            aspect_ratio=job.aspect_ratio,
            resolution=job.resolution,
        )
        job.request_id = response.request_id
        job.status = "processing"
        for _ in range(120):
            status = client.video.get(request_id=job.request_id)
            if getattr(status, "url", None):
                job.video_url = status.url
                job.status = "completed"
                return
            time.sleep(5)
        job.status = "error"
        job.error = "Timed out waiting for video URL"
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)


@app.get("/")
def index() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/iterate")
async def create_iteration(
    file: UploadFile = File(...),
    prompt: str = Form("")
) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat()
    job = IterateJob(id=job_id, prompt=prompt, created_at=created_at)
    with LOCK:
        ITERATE_JOBS[job_id] = job
    job_dir = STORAGE_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    r0_path = job_dir / "R0.png"
    r0_path.write_bytes(await file.read())

    thread = threading.Thread(target=_start_iteration, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/iterate/{job_id}")
def get_iteration(job_id: str) -> Dict[str, Any]:
    job = ITERATE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    candidates = {
        step: [
            {
                "id": candidate.id,
                "image_url": f"/files/{job.id}/{Path(candidate.image_path).name}",
                "description": candidate.description,
                "score": candidate.score,
            }
            for candidate in items
        ]
        for step, items in job.candidates.items()
    }

    thumbnails = {}
    for idx in range(0, job.step + 1):
        image_path = STORAGE_DIR / job.id / f"R{idx}.png"
        if image_path.exists():
            thumbnails[f"R{idx}"] = f"/files/{job.id}/R{idx}.png"

    return {
        "id": job.id,
        "prompt": job.prompt,
        "status": job.status,
        "step": job.step,
        "style_card": job.style_card,
        "candidates": candidates,
        "chosen": job.chosen,
        "final_image_url": f"/files/{job.id}/R3.png" if job.final_image_path else None,
        "thumbnails": thumbnails,
        "error": job.error,
    }


@app.post("/api/iterate/{job_id}/choose")
def choose_candidate(job_id: str, payload: ChooseRequest) -> Dict[str, Any]:
    job = ITERATE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "awaiting_choice":
        raise HTTPException(status_code=400, detail="Job not awaiting choice")

    step = job.step
    candidates = {c.id: c for c in job.candidates.get(step, [])}
    if payload.choice_id not in candidates:
        raise HTTPException(status_code=400, detail="Invalid choice")

    choice = candidates[payload.choice_id]
    dest = STORAGE_DIR / job.id / f"R{step}.png"
    Path(choice.image_path).replace(dest)
    job.chosen[step] = payload.choice_id

    if step == 3:
        job.final_image_path = str(dest)
        job.status = "completed"
    else:
        job.status = "ready_for_next"
    return {"status": job.status, "step": job.step}


@app.post("/api/iterate/{job_id}/next")
def next_step(job_id: str) -> Dict[str, Any]:
    job = ITERATE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in {"ready_for_next", "awaiting_choice"}:
        raise HTTPException(status_code=400, detail="Job not ready for next step")
    if job.step >= 3:
        raise HTTPException(status_code=400, detail="Already completed")
    if job.status == "awaiting_choice":
        raise HTTPException(status_code=400, detail="Choose a candidate first")

    next_step_index = job.step + 1
    thread = threading.Thread(
        target=_generate_candidates,
        args=(job, next_step_index),
        daemon=True,
    )
    thread.start()
    return {"status": "processing", "step": next_step_index}


@app.post("/api/video-jobs")
def create_video_job(payload: VideoJobRequest) -> Dict[str, Any]:
    if payload.duration < 1 or payload.duration > 15:
        raise HTTPException(status_code=400, detail="Duration must be 1-15 seconds")
    job_id = uuid.uuid4().hex
    job = VideoJob(
        id=job_id,
        status="queued",
        image_url=payload.image_url,
        prompt=payload.prompt,
        duration=payload.duration,
        aspect_ratio=payload.aspect_ratio,
        resolution=payload.resolution,
    )
    with LOCK:
        VIDEO_JOBS[job_id] = job
    thread = threading.Thread(target=_start_video_job, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/video-jobs/{job_id}")
def get_video_job(job_id: str) -> Dict[str, Any]:
    job = VIDEO_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "request_id": job.request_id,
        "video_url": job.video_url,
        "error": job.error,
    }


@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str) -> FileResponse:
    file_path = STORAGE_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
