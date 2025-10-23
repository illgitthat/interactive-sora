from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 fallback
    ConfigDict = None
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import init_db, session_scope
from models import (
    Scene,
    SceneMetric,
    SceneStatus,
    compute_depth,
    last_choice_index,
    parent_path,
    split_path,
)
from storage import LocalStorageClient, StoredAsset, build_storage_client
from admin import router as admin_router


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import imageio_ffmpeg  # type: ignore

    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG_BIN = None

DEFAULT_VIDEO_SIZE = "1280x720"
DEFAULT_SECONDS = 8
ALLOWED_SECONDS = (4, 8, 12)

WORLD_ID = os.environ.get("WORLD_ID", "default")

DEFAULT_WORLD_BASE_PROMPT = (
    "The Courier is an elite operative racing north through Manhattan from the Financial District toward the Upper West Side to secure chronoglyph shards before rival crews. "
    "Opening scene: in first-person, the Courier descends from a hovering stealth helicopter on a fast rope, feeling rotor wash and neon reflections off the glass canyons of FiDi. The rope slide ends with heavy boots hitting the rain-slick street, establishing an urgent foothold amid honking traffic, startled civilians, and distant gunfire. "
    "The mission begins the moment the player lands on the asphalt, weapon drawn and HUD flickering, ready to push through the maze of downtown streets toward the next objective."
)

BASE_PROMPT = os.environ.get("WORLD_BASE_PROMPT", DEFAULT_WORLD_BASE_PROMPT)
DEFAULT_PLANNER_MODEL = (
    os.environ.get("AZURE_OPENAI_CHAT_MODEL", "gpt-5-chat").strip() or "gpt-5-chat"
)
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip()
SORA_MODEL = os.environ.get("AZURE_OPENAI_SORA_MODEL", "sora-2").strip() or "sora-2"
VEO_MODEL = SORA_MODEL  # Backwards compatibility for response payloads
VIDEO_SIZE = (
    os.environ.get("VIDEO_SIZE", DEFAULT_VIDEO_SIZE).strip() or DEFAULT_VIDEO_SIZE
)
SCENE_TIMEOUT_SECONDS = int(os.environ.get("SCENE_TIMEOUT_SECONDS", "900"))
WATCHDOG_INTERVAL_SECONDS = int(os.environ.get("WATCHDOG_INTERVAL_SECONDS", "60"))
CONTRIBUTOR_SALT = os.environ.get("CONTRIBUTOR_SALT", "sora-shared-world")


def _max_concurrent_jobs() -> int:
    raw = os.getenv("SORA_MAX_CONCURRENT_VIDEO_JOBS", "2").strip() or "2"
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(1, value)


MAX_CONCURRENT_VIDEO_JOBS = _max_concurrent_jobs()
VIDEO_JOB_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_VIDEO_JOBS)

AZURE_API_BASE = os.getenv("AZURE_OPENAI_API_BASE", "").rstrip("/")
AZURE_RESPONSES_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "preview").strip()
AZURE_VIDEO_API_VERSION = (
    os.getenv("AZURE_OPENAI_VIDEO_API_VERSION", "").strip()
    or os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
)
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
APP_TITLE = "Sora Shared World API"

DEFAULT_PROMPT_GUIDANCE = "\n".join(
    [
        "Tone: Cinematic, gritty, high-adrenaline urban combat at dusk. Every scene emphasizes NYC landmarks or recognizable street-level details.",
        "Movement: Focus on dynamic, first-person action—running, taking cover, firing, reloading—maintaining intensity and realism.",
        "Environment: Depict NYC authentically but remixed by conflict (smoke, barricades, improvised covers, abandoned cars). Locations should reflect the route from Fidi toward the Upper West Side.",
        "Objective: Clearly show directional progress toward the Upper West Side with landmarks or street signs indicating northward movement.",
        "Allies & Foes: Encounters with rival groups, snipers, and unexpected combatants positioned strategically along the route. Highlight tactical maneuvers and exchanges of fire.",
        "Hook: Each scene ends with a sudden escalation (ambush, unexpected ally arrival, environmental hazard) compelling the next immediate decision or action.",
        "Checkpoint: Occasionally surface branching choices (alleys, rooftops, subway entrances) as immediate tactical decisions shaping the journey.",
    ]
)

PROMPT_GUIDANCE = (
    os.environ.get("WORLD_PROMPT_GUIDANCE", "").strip() or DEFAULT_PROMPT_GUIDANCE
)
STATE_SUMMARY_MODEL = os.environ.get(
    "STATE_SUMMARY_MODEL", DEFAULT_PLANNER_MODEL
).strip()

VIDEO_DIR = Path("sora_cyoa_videos")
FRAME_DIR = Path("sora_cyoa_frames")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
storage_client = build_storage_client()
LOG_LEVEL_NAME = os.getenv("SORA_LOG_LEVEL", "INFO").strip().upper() or "INFO"
logger = logging.getLogger("sora_shared_world")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL_NAME, logging.INFO))


def _looks_like_gemini_key(value: str) -> bool:
    if not value:
        return False
    gemini_prefixes = ("AIza", "AI", "gk-", "gw-")
    return any(value.startswith(prefix) for prefix in gemini_prefixes)


def _coalesce_planner_key(planner_key: str, fallback_key: str) -> str:
    key = (planner_key or "").strip()
    if key and _looks_like_gemini_key(key):
        return key
    if key and not _looks_like_gemini_key(key):
        logger.info(
            "[planner] supplied planner key does not resemble a Gemini key; falling back to video key"
        )
    return (fallback_key or "").strip()


app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if isinstance(storage_client, LocalStorageClient):
    app.mount(
        "/storage", StaticFiles(directory=storage_client.base_dir), name="storage"
    )

STATIC_DIR = Path("static")
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(admin_router)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    return response


class SceneResponse(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support

        class Config:
            allow_population_by_field_name = True

    world_id: str = Field(..., alias="worldId")
    path: str
    depth: int
    status: str
    scenario_display: Optional[str] = Field(None, alias="scenarioDisplay")
    veo_prompt: Optional[str] = Field(None, alias="veoPrompt")
    trigger_choice: Optional[str] = Field(None, alias="triggerChoice")
    choices: List[str] = Field(default_factory=list)
    choices_short: List[str] = Field(default_factory=list, alias="choicesShort")
    choices_status: List[str] = Field(default_factory=list, alias="choicesStatus")
    children_paths: List[str] = Field(default_factory=list, alias="childrenPaths")
    video_url: Optional[str] = Field(None, alias="videoUrl")
    poster_url: Optional[str] = Field(None, alias="posterUrl")
    failure_code: Optional[str] = Field(None, alias="failureCode")
    failure_detail: Optional[str] = Field(None, alias="failureDetail")
    queued_since: Optional[datetime] = Field(None, alias="queuedSince")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")
    progress: Optional[int] = None
    progress_updated_at: Optional[datetime] = Field(None, alias="progressUpdatedAt")
    state_summary: Optional[str] = Field(None, alias="stateSummary")
    context_video_seconds: Optional[int] = Field(None, alias="contextVideoSeconds")
    context_video_uri: Optional[str] = Field(None, alias="contextVideoUri")


class SceneGenerationRequest(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support

        class Config:
            allow_population_by_field_name = True

    path: str = ""
    planner_api_key: Optional[str] = Field(None, alias="plannerApiKey")
    video_api_key: Optional[str] = Field(None, alias="videoApiKey")
    api_key: Optional[str] = Field(None, alias="apiKey")

    @validator("path")
    def validate_path(cls, value: str) -> str:
        if value == "":
            return ""
        if not re.fullmatch(r"(\d+)(/\d+)*", value):
            raise ValueError("path must be slash-separated numeric indexes, e.g. '0/1'")
        return value


class WorldResponse(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support

        class Config:
            allow_population_by_field_name = True

    world_id: str = Field(..., alias="worldId")
    base_prompt: str = Field(..., alias="basePrompt")
    planner_model: str = Field(..., alias="plannerModel")
    veo_model: str = Field(..., alias="veoModel")
    video_size: str = Field(..., alias="videoSize")


class WorldMetricsResponse(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support

        class Config:
            allow_population_by_field_name = True

    world_id: str = Field(..., alias="worldId")
    scene_count: int = Field(..., alias="sceneCount")
    ready_count: int = Field(..., alias="readyCount")
    queued_count: int = Field(..., alias="queuedCount")
    failed_count: int = Field(..., alias="failedCount")
    storage_bytes: int = Field(..., alias="storageBytes")
    success_rate: float = Field(..., alias="successRate")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_seconds(secs: int) -> int:
    allowed = (4, 8, 12)
    return min(allowed, key=lambda value: abs(value - int(secs)))


@dataclass
class GenerationHandle:
    world_id: str
    path: str
    cancel_event: threading.Event
    thread: threading.Thread


_RUNNING_GENERATIONS: Dict[Tuple[str, str], GenerationHandle] = {}
_RUN_LOCK = threading.Lock()


def register_generation(handle: GenerationHandle) -> None:
    with _RUN_LOCK:
        _RUNNING_GENERATIONS[(handle.world_id, handle.path)] = handle


def clear_generation(world_id: str, path: str) -> None:
    with _RUN_LOCK:
        _RUNNING_GENERATIONS.pop((world_id, path), None)


def request_cancel(world_id: str, path: str) -> None:
    with _RUN_LOCK:
        handle = _RUNNING_GENERATIONS.get((world_id, path))
        if handle:
            handle.cancel_event.set()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    threading.Thread(target=_timeout_watchdog, daemon=True).start()


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/worlds/{world_id}", response_model=WorldResponse)
def get_world(world_id: str) -> WorldResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")
    return WorldResponse(
        worldId=WORLD_ID,
        basePrompt=BASE_PROMPT,
        plannerModel=PLANNER_MODEL,
        veoModel=VEO_MODEL,
        videoSize=VIDEO_SIZE,
    )


@app.get("/worlds/{world_id}/metrics", response_model=WorldMetricsResponse)
def get_world_metrics(world_id: str) -> WorldMetricsResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")

    with session_scope() as session:
        total = session.execute(
            select(func.count()).where(Scene.world_id == world_id)
        ).scalar_one()
        ready = session.execute(
            select(func.count()).where(
                Scene.world_id == world_id, Scene.status == SceneStatus.READY
            )
        ).scalar_one()
        queued = session.execute(
            select(func.count()).where(
                Scene.world_id == world_id, Scene.status == SceneStatus.QUEUED
            )
        ).scalar_one()
        failed = session.execute(
            select(func.count()).where(
                Scene.world_id == world_id, Scene.status == SceneStatus.FAILED
            )
        ).scalar_one()
        storage_bytes = session.execute(
            select(func.coalesce(func.sum(SceneMetric.storage_bytes), 0))
            .join(Scene, SceneMetric.scene_id == Scene.id)
            .where(Scene.world_id == world_id)
        ).scalar_one()

    success_denominator = max(ready + failed, 1)
    success_rate = ready / success_denominator

    return WorldMetricsResponse(
        worldId=world_id,
        sceneCount=total,
        readyCount=ready,
        queuedCount=queued,
        failedCount=failed,
        storageBytes=int(storage_bytes or 0),
        successRate=success_rate,
    )


@app.get("/worlds/{world_id}/scenes", response_model=SceneResponse)
def get_scene(world_id: str, path: str = Query("")) -> SceneResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")

    with session_scope() as session:
        scene = ensure_scene_exists(session, world_id, path)
        return build_scene_response(session, scene)


@app.post("/worlds/{world_id}/scenes", response_model=SceneResponse)
def generate_scene_endpoint(
    world_id: str, payload: SceneGenerationRequest
) -> SceneResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")

    path = payload.path or ""
    legacy_key = (payload.api_key or "").strip()
    video_api_key = (payload.video_api_key or legacy_key).strip()
    planner_api_key = _coalesce_planner_key(
        payload.planner_api_key, video_api_key or legacy_key
    )

    if not video_api_key:
        raise HTTPException(
            status_code=400, detail="Video API key required for generation"
        )
    if not planner_api_key:
        planner_api_key = video_api_key

    should_start = False
    with session_scope() as session:
        scene = ensure_scene_exists(session, world_id, path)
        if scene.status == SceneStatus.READY:
            logger.info(
                "scene already ready world=%s path=%s", world_id, path or "root"
            )
            return build_scene_response(session, scene)
        if scene.status == SceneStatus.QUEUED:
            logger.info(
                "scene already queued world=%s path=%s", world_id, path or "root"
            )
            return build_scene_response(session, scene)

        # pending or failed
        logger.info("scene claim queued world=%s path=%s", world_id, path or "root")
        scene.status = SceneStatus.QUEUED
        scene.failure_code = None
        scene.failure_detail = None
        scene.started_at = utcnow()
        scene.progress = 0
        scene.progress_updated_at = utcnow()
        session.flush()
        should_start = True
        response = build_scene_response(session, scene)

    if should_start:
        start_generation(world_id, path, planner_api_key, video_api_key)
    return response


@app.post("/worlds/{world_id}/scenes/{path:path}/retry", response_model=SceneResponse)
def retry_scene(
    world_id: str, path: str, payload: SceneGenerationRequest
) -> SceneResponse:
    payload.path = path
    return generate_scene_endpoint(world_id, payload)


def ensure_scene_exists(session: Session, world_id: str, path: str) -> Scene:
    stmt = select(Scene).where(Scene.world_id == world_id, Scene.path == path)
    scene = session.execute(stmt).scalars().first()
    if scene:
        return scene

    scene = Scene(
        world_id=world_id,
        path=path,
        depth=compute_depth(path),
        status=SceneStatus.PENDING,
    )
    parent = parent_path(path)
    if parent is not None:
        scene.trigger_choice = None
    session.add(scene)
    session.flush()
    return scene


def _resolve_asset_url(value: Optional[str], *, variant: str) -> Optional[str]:
    if not value:
        return None
    try:
        return storage_client.resolve_url(value, variant=variant)
    except AttributeError:
        # Back-compat for older StorageClient implementations
        return value


def _update_scene_progress(world_id: str, path: str, progress: Optional[int]) -> None:
    if progress is None:
        return
    with session_scope() as session:
        scene = (
            session.execute(
                select(Scene).where(Scene.world_id == world_id, Scene.path == path)
            )
            .scalars()
            .first()
        )
        if not scene or scene.status != SceneStatus.QUEUED:
            return
        scene.progress = int(progress)
        scene.progress_updated_at = utcnow()


def ensure_action_beat(scene: Dict[str, Any], fallback_choice: Optional[str]) -> None:
    prompt = scene.get("veo_prompt") or ""
    if "Action Beat:" in prompt:
        logger.info("[prompt] action beat already present")
        return
    candidate = fallback_choice or ""
    if not candidate:
        choices = scene.get("choices") or []
        if choices:
            candidate = choices[0]
        else:
            candidate = (scene.get("scenario_display") or "")[:160]
    candidate = candidate.strip()
    if not candidate:
        candidate = "Trigger a dramatic cross-world portal event within 8 seconds."
    scene["veo_prompt"] = prompt.rstrip() + f"\nAction Beat: {candidate}"
    logger.info("[prompt] appended action beat: %s", candidate)


def build_scene_response(session: Session, scene: Scene) -> SceneResponse:
    choices = scene.choices if isinstance(scene.choices, list) else []
    choices_short = scene.choices_short if isinstance(scene.choices_short, list) else []
    if not choices_short:
        choices_short = list(choices)
    else:
        normalized_short: List[str] = []
        for idx in range(len(choices)):
            short_value = choices_short[idx] if idx < len(choices_short) else None
            long_value = choices[idx] if idx < len(choices) else None
            candidate = (short_value or "").strip()
            if not candidate and long_value is not None:
                candidate = str(long_value).strip()
            if not candidate:
                candidate = f"Choice {idx + 1}"
            normalized_short.append(candidate)
        choices_short = normalized_short
    child_statuses: List[str] = []
    child_paths: List[str] = []
    for idx in range(len(choices) or 3):
        child = (
            scene.child_path(idx)
            if hasattr(scene, "child_path")
            else _child_path(scene.path, idx)
        )
        child_paths.append(child)
        child_scene = (
            session.execute(
                select(Scene).where(
                    Scene.world_id == scene.world_id, Scene.path == child
                )
            )
            .scalars()
            .first()
        )
        if child_scene is None:
            child_statuses.append(SceneStatus.PENDING.value)
        else:
            child_statuses.append(child_scene.status.value)

    return SceneResponse(
        worldId=scene.world_id,
        path=scene.path,
        depth=scene.depth,
        status=scene.status.value,
        scenarioDisplay=scene.scenario_display,
        veoPrompt=scene.veo_prompt,
        triggerChoice=scene.trigger_choice,
        choices=choices,
        choices_short=choices_short,
        choicesStatus=child_statuses,
        childrenPaths=child_paths,
        videoUrl=_resolve_asset_url(scene.video_url, variant="video"),
        posterUrl=_resolve_asset_url(scene.poster_url, variant="poster"),
        failureCode=scene.failure_code,
        failureDetail=scene.failure_detail,
        queuedSince=scene.started_at,
        updatedAt=scene.updated_at,
        progress=getattr(scene, "progress", None),
        progressUpdatedAt=getattr(scene, "progress_updated_at", None),
        stateSummary=getattr(scene, "state_summary", None),
        contextVideoSeconds=getattr(scene, "context_video_seconds", None),
        contextVideoUri=getattr(scene, "context_video_uri", None),
    )


def start_generation(
    world_id: str, path: str, planner_api_key: str, video_api_key: str
) -> None:
    cancel_event = threading.Event()
    thread = threading.Thread(
        target=_generate_scene,
        args=(world_id, path, planner_api_key, video_api_key, cancel_event),
        daemon=True,
        name=f"gen-{world_id}-{path or 'root'}",
    )
    handle = GenerationHandle(
        world_id=world_id, path=path, cancel_event=cancel_event, thread=thread
    )
    register_generation(handle)
    logger.info("generation queued world=%s path=%s", world_id, path or "root")
    thread.start()


def _generate_scene(
    world_id: str,
    path: str,
    planner_api_key: str,
    video_api_key: str,
    cancel_event: threading.Event,
) -> None:
    contributor_hash = hash_contributor(video_api_key, path)
    try:
        logger.info("generation started world=%s path=%s", world_id, path or "root")
        try:
            _generate_scene_inner(
                world_id,
                path,
                planner_api_key,
                video_api_key,
                cancel_event,
                contributor_hash,
            )
        except SceneCancelled:
            logger.info(
                "generation cancelled world=%s path=%s", world_id, path or "root"
            )
            _mark_pending(world_id, path)
        except Exception as exc:
            logger.exception(
                "generation error world=%s path=%s", world_id, path or "root"
            )
            _mark_failed(world_id, path, "generation_error", str(exc))
    finally:
        logger.info("generation finished world=%s path=%s", world_id, path or "root")
        clear_generation(world_id, path)


def _generate_scene_inner(
    world_id: str,
    path: str,
    planner_api_key: str,
    video_api_key: str,
    cancel_event: threading.Event,
    contributor_hash: str,
) -> None:
    with session_scope() as session:
        scene = (
            session.execute(
                select(Scene)
                .where(Scene.world_id == world_id, Scene.path == path)
                .with_for_update()
            )
            .scalars()
            .one()
        )
        if scene.status != SceneStatus.QUEUED:
            return
        scene.started_at = scene.started_at or utcnow()
        session.flush()

    if cancel_event.is_set():
        _mark_pending(world_id, path)
        return

    planner_result = plan_scene(world_id, path, planner_api_key)
    if planner_result.get("_planner_missing_prompt"):
        _mark_failed(
            world_id,
            path,
            "planner_missing_prompt",
            planner_result.get("_planner_missing_prompt_reason", ""),
        )
        return

    if cancel_event.is_set():
        _mark_pending(world_id, path)
        return

    ensure_action_beat(planner_result, planner_result.get("_chosen_choice"))

    try:
        asset, new_context_seconds, context_video_uri = render_scene_video(
            world_id,
            path,
            planner_result.get("sora_prompt") or planner_result["veo_prompt"],
            video_api_key,
            cancel_event,
        )
    except SceneCancelled:
        _mark_pending(world_id, path)
        return
    except Exception as exc:
        _mark_failed(world_id, path, "veo_error", str(exc))
        return

    if cancel_event.is_set():
        _mark_pending(world_id, path)
        return

    prior_state_summaries = collect_state_summaries(world_id, path)
    state_summary_text = summarise_scene_state(
        api_key=planner_api_key,
        base_prompt=BASE_PROMPT,
        scenario_display=planner_result["scenario_display"],
        choices=planner_result["choices"],
        prior_summaries=prior_state_summaries,
    )
    if not state_summary_text:
        state_summary_text = planner_result["scenario_display"]

    with session_scope() as session:
        scene = (
            session.execute(
                select(Scene)
                .where(Scene.world_id == world_id, Scene.path == path)
                .with_for_update()
            )
            .scalars()
            .one()
        )
        scene.scenario_display = planner_result["scenario_display"]
        scene.veo_prompt = planner_result["veo_prompt"]
        scene.choices = planner_result["choices"]
        scene.choices_short = planner_result.get("choices_short")
        scene.planner_model = PLANNER_MODEL
        scene.planner_raw = planner_result.get("_raw_planner_output")
        if isinstance(storage_client, LocalStorageClient):
            scene.video_url = asset.video_url
            scene.poster_url = asset.poster_url
            scene.context_video_url = asset.context_video_url
        else:
            scene.video_url = asset.video_key
            scene.poster_url = asset.poster_key
            scene.context_video_url = asset.context_video_key
        scene.video_seconds = DEFAULT_SECONDS
        scene.context_video_seconds = new_context_seconds
        scene.context_video_uri = context_video_uri
        scene.status = SceneStatus.READY
        scene.failure_code = None
        scene.failure_detail = None
        scene.contributor_hash = contributor_hash
        scene.started_at = None
        scene.state_summary = state_summary_text
        scene.progress = 100
        scene.progress_updated_at = utcnow()
        scene.trigger_choice = determine_trigger_choice(session, world_id, path)
        session.add(
            SceneMetric(
                scene_id=scene.id,
                rendered=1,
                render_time_ms=None,
                storage_bytes=asset.bytes_written,
            )
        )


def determine_trigger_choice(
    session: Session, world_id: str, path: str
) -> Optional[str]:
    parent = parent_path(path)
    if parent is None:
        return None
    parent_scene = (
        session.execute(
            select(Scene).where(Scene.world_id == world_id, Scene.path == parent)
        )
        .scalars()
        .first()
    )
    if parent_scene is None or not parent_scene.choices:
        return None
    idx = last_choice_index(path)
    if idx is None:
        return None
    if idx < len(parent_scene.choices):
        return parent_scene.choices[idx]
    return None


def plan_scene(world_id: str, path: str, api_key: str) -> Dict[str, Any]:
    if not path:
        result = plan_initial_scene(
            api_key=api_key, base_prompt=BASE_PROMPT, model=PLANNER_MODEL
        )
        first_choice = (result.get("choices") or [None])[0]
        result["_chosen_choice"] = first_choice
        result["_state_context"] = []
    else:
        parent_path_value = parent_path(path)
        if parent_path_value is None:
            raise RuntimeError("Path has no parent; cannot continue")
        ancestor_paths = ancestor_path_list(path)
        with session_scope() as session:
            stmt = select(Scene).where(
                Scene.world_id == world_id, Scene.path.in_(ancestor_paths)
            )
            rows = session.execute(stmt).scalars().all()
        by_path = {row.path: row for row in rows}
        parent = by_path.get(parent_path_value)
        if parent is None or not parent.choices:
            raise RuntimeError("Parent scene lacks choices; cannot continue")
        prior_prompts = []
        state_context: List[str] = []
        for anc_path in ancestor_paths:
            scene = by_path.get(anc_path)
            if scene and scene.veo_prompt:
                prior_prompts.append(scene.veo_prompt)
            if anc_path != path and scene and getattr(scene, "state_summary", None):
                state_context.append(scene.state_summary)
        idx = last_choice_index(path)
        if idx is None or idx >= len(parent.choices):
            raise RuntimeError("Invalid choice index for path")
        chosen_choice = parent.choices[idx]
        logger.info(
            "[planner] continue world=%s path=%s choice=%s state_context=%s",
            world_id,
            path,
            chosen_choice,
            state_context,
        )
        result = plan_next_scene(
            api_key=api_key,
            base_prompt=BASE_PROMPT,
            prior_video_prompts=prior_prompts,
            chosen_choice=chosen_choice,
            state_summaries=state_context,
            model=PLANNER_MODEL,
        )
        result["_chosen_choice"] = chosen_choice
        result["_state_context"] = state_context
    return result


class SceneCancelled(Exception):
    pass


def ancestor_path_list(path: str) -> List[str]:
    parts = split_path(path)
    ancestors: List[str] = []
    for end in range(1, len(parts) + 1):
        ancestor = "/".join(str(part) for part in parts[:end])
        ancestors.append(ancestor)
    if ancestors:
        # Always include root path "" as the base context
        ancestors.insert(0, "")
    else:
        ancestors.append("")
    return ancestors


def collect_state_summaries(world_id: str, path: str) -> List[str]:
    ancestor_paths = ancestor_path_list(path)
    # Exclude the current path; we only need previously locked scenes
    ancestor_context = [p for p in ancestor_paths if p != path]
    if not ancestor_context:
        return []
    with session_scope() as session:
        rows = (
            session.execute(
                select(Scene).where(
                    Scene.world_id == world_id, Scene.path.in_(ancestor_context)
                )
            )
            .scalars()
            .all()
        )
    rows.sort(key=lambda scene: scene.depth)
    summaries = [
        row.state_summary for row in rows if getattr(row, "state_summary", None)
    ]
    return summaries


STATE_SUMMARY_SYSTEM = """
You are the chronicler for an expansive multiverse adventure.

Summarise the current state in at most three short bullet points.
- Track key elements: chronoglyph shards remaining/found, portal stability, allies or foes involved, immediate threats, and location shifts between worlds.
- Highlight cause and effect (e.g., how actions in one world impact another).
- Keep bullets under 160 characters, starting each with "- ". No extra commentary.
""".strip()


def summarise_scene_state(
    api_key: str,
    base_prompt: str,
    scenario_display: str,
    choices: List[str],
    prior_summaries: List[str],
) -> Optional[str]:
    model = STATE_SUMMARY_MODEL or ""
    if not model or model.lower() == "none":
        return None

    prior_section = (
        "\n".join(f"- {summary}" for summary in prior_summaries)
        if prior_summaries
        else "(none yet)"
    )
    choices_section = "\n".join(f"- {choice}" for choice in choices)
    user_input = f"""
WORLD BASE PROMPT (trimmed):
{base_prompt[:800]}

PRIOR STATE SNAPSHOT:
{prior_section}

CURRENT SCENE NARRATION:
{scenario_display}

CHOICES OFFERED NEXT:
{choices_section}

TASK: Summarise the evolving state using at most three bullets as instructed.
""".strip()

    try:
        summary_text = responses_create(
            api_key=api_key,
            model=model,
            instructions=STATE_SUMMARY_SYSTEM,
            user_input=user_input,
        )
        cleaned = summary_text.strip()
        return cleaned if cleaned else None
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("state summary generation failed: %s", exc)
        return None


def render_scene_video(
    world_id: str,
    path: str,
    sora_prompt: str,
    api_key: str,
    cancel_event: threading.Event,
) -> Tuple[StoredAsset, int, Optional[str]]:
    parent_context_path: Optional[Path] = None
    parent_context_uri: Optional[str] = None
    parent = parent_path(path)
    if parent is not None:
        with session_scope() as session:
            parent_scene = (
                session.execute(
                    select(Scene).where(
                        Scene.world_id == world_id, Scene.path == parent
                    )
                )
                .scalars()
                .first()
            )
        if parent_scene:
            logger.info(
                "[continuity] parent scene world=%s parent_path=%s status=%s",
                world_id,
                parent,
                getattr(parent_scene, "status", None),
            )
            source_value: Optional[str] = None
            variant = "context"
            if parent_scene.context_video_url:
                source_value = parent_scene.context_video_url
            elif parent_scene.video_url:
                source_value = parent_scene.video_url
                variant = "video"
            if source_value:
                parent_context_path = download_asset(source_value, variant=variant)
                if not (
                    isinstance(parent_context_path, Path)
                    and parent_context_path.exists()
                ):
                    parent_context_path = None
            stored_total = getattr(parent_scene, "context_video_seconds", None)
            if stored_total is None:
                stored_total = getattr(parent_scene, "video_seconds", None)
            parent_context_uri = getattr(parent_scene, "context_video_uri", None)
        else:
            logger.warning(
                "[continuity] missing parent scene world=%s parent_path=%s",
                world_id,
                parent,
            )

    input_reference = None
    if parent_context_uri:
        input_reference = parent_context_path
    elif parent_context_path and parent_context_path.exists():
        input_reference = parent_context_path

    asset: Optional[StoredAsset] = None
    context_video_uri: Optional[str] = None
    new_context_seconds = DEFAULT_SECONDS

    try:
        logger.info(
            "[sora] starting generation world=%s path=%s with context=%s",
            world_id,
            path or "root",
            bool(input_reference),
        )
        video_id, video_path, poster_path = generate_scene_video(
            api_key=api_key,
            sora_prompt=sora_prompt,
            model=SORA_MODEL,
            size=VIDEO_SIZE,
            seconds=DEFAULT_SECONDS,
            input_reference=input_reference
            if isinstance(input_reference, Path)
            else None,
        )
        if cancel_event.is_set():
            raise SceneCancelled()

        measured_duration = _video_duration_seconds(video_path) or DEFAULT_SECONDS
        new_context_seconds = int(math.ceil(measured_duration))

        key_prefix = f"{world_id}/{path or 'root'}"
        asset = storage_client.upload(
            video_path,
            poster_path,
            key_prefix=key_prefix,
            context_video_path=video_path,
        )
        logger.info(
            "[sora] uploaded asset key_prefix=%s video=%s poster=%s context=%s",
            key_prefix,
            asset.video_url,
            asset.poster_url,
            asset.context_video_url,
        )
    finally:
        if parent_context_path and parent_context_path.exists():
            parent_context_path.unlink(missing_ok=True)

    if asset is None:
        raise RuntimeError("Sora generation failed: no asset produced")
    return asset, new_context_seconds, context_video_uri


def download_asset(stored_value: str, variant: str) -> Optional[Path]:
    resolved_url = _resolve_asset_url(stored_value, variant=variant)
    if not resolved_url:
        logger.warning(
            "[continuity] resolve failed for variant=%s value=%s", variant, stored_value
        )
        return None
    logger.info(
        "[continuity] downloading asset variant=%s url=%s", variant, resolved_url
    )
    try:
        response = requests.get(resolved_url, timeout=30)
        logger.info(
            "[continuity] download status=%s url=%s", response.status_code, resolved_url
        )
        if response.status_code >= 400:
            logger.warning(
                "[continuity] download failed status=%s body=%s",
                response.status_code,
                response.text[:200],
            )
            return None
        parsed = urlparse(resolved_url)
        path_suffix = Path(parsed.path).suffix.lower()
        if path_suffix in {".jpg", ".jpeg", ".png", ".webp", ".mp4"}:
            suffix = path_suffix
        else:
            if variant in {"video", "context"}:
                suffix = ".mp4"
            else:
                suffix = ".mp4" if variant == "video" else ".jpg"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp = Path(tmp_path)
        with tmp.open("wb") as fh:
            fh.write(response.content)
        logger.info("[continuity] download saved to %s", tmp)
        return tmp
    except Exception as exc:
        logger.warning("[continuity] exception downloading asset: %s", exc)
        return None


def hash_contributor(api_key: str, path: str) -> str:
    import hashlib

    payload = f"{CONTRIBUTOR_SALT}:{path}:{api_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mark_pending(world_id: str, path: str) -> None:
    with session_scope() as session:
        scene = (
            session.execute(
                select(Scene).where(Scene.world_id == world_id, Scene.path == path)
            )
            .scalars()
            .first()
        )
        if not scene:
            return
        scene.status = SceneStatus.PENDING
        scene.started_at = None
        scene.failure_code = None
        scene.failure_detail = None
        scene.contributor_hash = None
        scene.progress = None
        scene.progress_updated_at = None
        logger.info("scene reset to pending world=%s path=%s", world_id, path or "root")


def _mark_failed(world_id: str, path: str, code: str, detail: str) -> None:
    with session_scope() as session:
        scene = (
            session.execute(
                select(Scene).where(Scene.world_id == world_id, Scene.path == path)
            )
            .scalars()
            .first()
        )
        if not scene:
            return
        scene.status = SceneStatus.FAILED
        scene.failure_code = code
        scene.failure_detail = detail
        scene.started_at = None
        scene.contributor_hash = None
        scene.progress = None
        scene.progress_updated_at = None
        logger.warning(
            "scene failed world=%s path=%s code=%s detail=%s",
            world_id,
            path or "root",
            code,
            detail,
        )


def _timeout_watchdog() -> None:
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        cutoff = utcnow() - timedelta(seconds=SCENE_TIMEOUT_SECONDS)
        try:
            with session_scope() as session:
                stmt = select(Scene).where(
                    Scene.world_id == WORLD_ID,
                    Scene.status == SceneStatus.QUEUED,
                    Scene.started_at.isnot(None),
                )
                rows = session.execute(stmt).scalars().all()
                for scene in rows:
                    if scene.started_at and scene.started_at < cutoff:
                        request_cancel(scene.world_id, scene.path)
                        scene.status = SceneStatus.PENDING
                        scene.started_at = None
        except Exception:
            continue


def _child_path(path: str, index: int) -> str:
    if not path:
        return str(index)
    return f"{path}/{index}"


# === Planner Helpers ===

PLANNER_SYSTEM = """
You are the Scenario Planner for an Azure OpenAI Sora-powered choose-your-own-adventure game.

Your job:
- Given a BASE PROMPT (world/tone) or a CONTINUATION (previous scene prompts + the player's chosen action),
- Produce a JSON object that contains:
  {
    "scenario_display": "A short paragraph (<= 120 words) narrating the current scene to show in the UI.",
    "sora_prompt": "<A detailed Sora prompt for generating an 8-second extension segment.>",
    "choices": ["<choice 1>", "<choice 2>", "<choice 3>"],
    "choices_short": ["<concise choice 1>", "<concise choice 2>", "<concise choice 3>"]
  }

Great Example (match this intensity & structure):
{
  "scenario_display": "You rocket up the slick fire escape, vault the parapet, and rip a two-round burst that sends the rooftop sniper sliding toward a neon UPTOWN arrow. Drone rotors howl, a stairwell door slams open with a ticking flashbang, and the shard beacon flares cyan over the next roof. Five seconds before they box you in—choose fast.",
  "sora_prompt": "Context (not visible in video, only for AI guidance): First-person courier sprinting north across Broadway rooftops seconds after breaking taxi cover; rival drone stalking from the east, stairwell squad erupting behind.\nAction 0-2s: Whip right through deli steam, grapple the fire escape, yank it down, and rocket skyward three rungs at a time.\nAction 2-5s: Vault the parapet, skid across rain-slick tar, rip a suppressed double tap that sends the sniper’s rifle skittering past a BROADWAY sign.\nAction 5-8s: Drop into a knee slide toward a glowing UPTOWN ladder as a quadcopter strafes sparks and the stair door detonates showering bolts.\nCamera: Head-cam surges handheld; whip pan left to the sniper hit, then slam low tracking the ladder sprint.\nAudio: Sirens Doppler below, rotor whine screams overhead, flashbang pin clinks, HUD bleeps urgent countdowns.\nFX: Rain beads, muzzle flares, steam bursts, cyan chronoglyph shimmer pulses over the 44th St water tower.\nMomentum North: Drive toward the 44th St tower beacon within 8 seconds before rival scouts seize the shard route.",
  "choices": [
    "Command: Leap the UPTOWN ladder and sprint the water-tower catwalk (Risk: exposed to drone fire; Payoff: seize overwatch on the shard beacon).",
    "Command: Slam the stairwell door shut and plant a flash trap (Risk: CQB slugfest; Payoff: wipe the pursuers and steal their uplink codes).",
    "Command: Snatch the drone mid-air and ride it toward Fulton Street (Risk: mid-air vulnerability; Payoff: rocket three blocks north in under 6 seconds)."
  ],
  "choices_short": [
    "Take ladder—own the beacon",
    "Trap the stair squad",
    "Hijack drone to Fulton"
  ]
}

Weak Example (avoid this):
{
  "scenario_display": "You run beside the bus and shoot at enemies. The bus stops and civilians scream. You must decide what to do next.",
  "sora_prompt": "Context: You are on Broadway.\nAction 0-2s: Run.\nAction 2-5s: Shoot.\nAction 5-8s: Take cover.\nCamera: Follow the player.\nAudio: City noise.\nFX: Rain.\nMomentum North: Keep going north.",
  "choices": [
    "Command: Help the civilians.",
    "Command: Keep shooting.",
    "Command: Hide inside the bus."
  ],
  "choices_short": ["Help civilians", "Keep firing", "Hide in bus"]
}

Rules:
1) The 'sora_prompt' must be the exact text we send to Azure OpenAI Sora.
   - Begin with "Context (not visible in video, only for AI guidance): ..." to recap continuity and constraints.
   - Follow with the lines below **exactly in this order**, each written as present-tense fragments overflowing with kinetic verbs (vault, crash, rip, rocket, detonate, grapple, etc.). Avoid tame verbs like "move" or "look" unless paired with something explosive.
       * "Action 0-2s: ..."
       * "Action 2-5s: ..."
       * "Action 5-8s: ..."
       * "Camera: ..."
       * "Audio: ..."
       * "FX: ..."
       * "Momentum North: ..." (explicitly state the next NYC landmark, street, or chronoglyph vector you’re driving toward and how this beat accelerates the courier north within seconds.)
   - Each line must stay ≤ 22 words and jammed with layered sensory detail (motion + location + tactile + objective).
   - Assume the engine feeds Sora the full prior video, so design seamless momentum across cuts (matching motion, camera, props).

2) Safety & platform constraints (strict):
   - Content must be suitable for audiences under 18.
   - Do NOT depict real people (including public figures) or copyrighted/fictional characters.
   - Avoid copyrighted music and explicit logos/trademarks. Use generic brand cues only.
   - Avoid hate, sexual content, excessive violence, or self-harm.

3) Continuity:
   - Maintain consistent characters, setting, tone, camera language, and lighting unless the choice implies a justified shift.
   - Ensure smooth shot-to-shot transitions; the new beat should feel like the same take continuing from the previous clip.

4) Choices:
   - Provide exactly three distinct options for what the player can do next.
   - Format each entry as "Command: <high-velocity action> (Risk: <danger>; Payoff: <mission gain>)" so risk/reward is unmistakable.
   - Keep each entry ≤ 22 words, packed with muscular verbs, and ensure all three actions diverge dramatically in direction or approach (vertical vs. subterranean vs. vehicular, etc.).
   - Provide a matching `choices_short` array: same order, each entry ≤ 12 words, phrased as urgent imperatives that hint at the payoff.

5) Scenario essentials (aligned to BASE_PROMPT and PROMPT_GUIDANCE):
   - Tone: Cinematic, gritty, high-adrenaline urban combat at dusk. Scenes emphasize NYC landmarks or recognizable street-level details.
   - Movement: Dynamic, first-person action (running, taking cover, firing, reloading), intensity, and realism maintained.
   - Environment: Authentic but conflict-remixed NYC (smoke, barricades, improvised cover, abandoned vehicles). Locations reflect the route from FiDi toward Upper West Side.
   - Objective: Clearly show directional progress northward toward the Upper West Side, using landmarks or street signs.
   - Allies & Foes: Rival operatives, snipers, unexpected combatants strategically positioned. Highlight tactical maneuvers and exchanges of fire.
   - Hook: End scenes with sudden escalations (ambush, unexpected ally arrival, environmental hazard), compelling immediate next actions.
   - Checkpoint: Occasionally provide branching tactical decisions (alleys, rooftops, subway entrances) shaping journey and outcomes.

6) Pacing & shot design:
   - Each 8-second beat must deliver a complete moment (setup → escalation → visible outcome) that meaningfully changes the situation.
   - Start in motion—no static openings. Smash into the beat mid-sprint or mid-impact within the first second.
   - End with a fresh reveal, reaction, or consequence that forces the next immediate choice.

7) Momentum & sensory cues:
   - Keep urgency palpable: layer in aggressive verbs, snap decisions, sprinting chases, collisions, or close calls every beat.
   - Thread in micro-stakes (e.g., dwindling ammo, civilians in the crossfire, rival squads closing in) so tension keeps rising.
   - Ban languid words like "pauses", "calm", or "regains composure". If a beat momentarily slows, frame it as a held-breath twitch before the next explosive move.
   - Ensure "Action" lines, "Camera", "Audio", "FX", and "Momentum North" braid together—every camera move should amplify the physical motion, soundscape, and forward objective.

8) Scenario_display tone:
   - Narration should read like a breathless field report under fire, with at least three high-velocity verbs (vaults, ricochets, detonates, shreds, etc.).
   - Always name the northbound landmark or chronoglyph vector that’s about to be seized and the ticking threat that will detonate/escalate within seconds if the player hesitates.
   - End on a cliffhanger sentence that screams for an immediate decision (“Five seconds before the barricade seals—move.”).

9) Output strictly JSON. No markdown, no commentary, no code fences.
""".strip()


def extract_first_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Planner did not return JSON. Received:\n" + text[:800])
    return json.loads(match.group(0))


def normalize_scene_payload(scene: Dict[str, Any]) -> Dict[str, Any]:
    def _pick(keys: List[str]) -> Any:
        for key in keys:
            if key in scene:
                value = scene[key]
                if value is None:
                    continue
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                return value
        return None

    scenario_display_keys = [
        "scenario_display",
        "scene_display",
        "scene_description",
        "scenario_description",
        "narration",
        "description",
        "display",
        "story",
    ]
    veo_prompt_keys = [
        "veo_prompt",
        "veoPrompt",
        "sora_prompt",
        "soraPrompt",
        "prompt",
        "video_prompt",
        "videoPrompt",
        "scene_prompt",
        "scenePrompt",
        "shot_prompt",
        "shotPrompt",
    ]
    choices_keys = ["choices", "options", "next_choices", "actions", "nextOptions"]
    choices_short_keys = [
        "choices_short",
        "choicesShort",
        "concise_choices",
        "conciseChoices",
        "short_choices",
        "shortChoices",
    ]

    scenario_display = _pick(scenario_display_keys)
    if isinstance(scenario_display, list):
        parts = [str(item).strip() for item in scenario_display if str(item).strip()]
        scenario_display = " ".join(parts)
    if not scenario_display:
        scenario_display = (
            "Planner response missing scene description. Adjust your prompt and retry."
        )

    veo_prompt_raw = _pick(veo_prompt_keys)
    veo_prompt_missing = False
    veo_prompt_missing_reason = ""

    veo_prompt_value: Any = veo_prompt_raw
    if isinstance(veo_prompt_value, dict):
        lines: List[str] = []
        for key, value in veo_prompt_value.items():
            if value is None:
                continue
            text_val = str(value).strip()
            if not text_val:
                continue
            lines.append(f"{key}: {text_val}")
        if lines:
            veo_prompt_value = "\n".join(lines).strip()
        else:
            veo_prompt_missing = True
            veo_prompt_missing_reason = (
                "Planner returned prompt dict but it had no usable values."
            )
            veo_prompt_value = ""
    elif isinstance(veo_prompt_value, list):
        joined = "\n".join(
            str(item).strip() for item in veo_prompt_value if str(item).strip()
        )
        if joined:
            veo_prompt_value = joined
        else:
            veo_prompt_missing = True
            veo_prompt_missing_reason = (
                "Planner returned prompt list but all entries were empty."
            )
            veo_prompt_value = ""

    if veo_prompt_value is None:
        veo_prompt_missing = True
        if not veo_prompt_missing_reason:
            veo_prompt_missing_reason = (
                "Planner response missing recognized Sora prompt field."
            )
        veo_prompt_value = ""
    elif not isinstance(veo_prompt_value, str):
        veo_prompt_value = str(veo_prompt_value).strip()
        if not veo_prompt_value:
            veo_prompt_missing = True
            if not veo_prompt_missing_reason:
                veo_prompt_missing_reason = (
                    "Planner returned non-string prompt that was empty after casting."
                )
    else:
        veo_prompt_value = veo_prompt_value.strip()
        if not veo_prompt_value:
            veo_prompt_missing = True
            if not veo_prompt_missing_reason:
                veo_prompt_missing_reason = "Planner Sora prompt string was blank."

    veo_prompt = (
        "Planner response missing Sora prompt details. Please tweak your base prompt or retry."
        if veo_prompt_missing
        else veo_prompt_value
    )

    raw_choices = _pick(choices_keys)
    choices: List[str] = []
    if isinstance(raw_choices, list):
        choices = [str(choice).strip() for choice in raw_choices if str(choice).strip()]
    elif isinstance(raw_choices, str):
        fragments = re.split(r"[\n|]", raw_choices)
        choices = [frag.strip(" •-\t").strip() for frag in fragments if frag.strip()]

    while len(choices) < 3:
        choices.append(
            f"Missing choice {len(choices) + 1}. Update prompt and regenerate."
        )
    if len(choices) > 3:
        choices = choices[:3]

    raw_choices_short = _pick(choices_short_keys)
    choices_short: List[str] = []
    if isinstance(raw_choices_short, list):
        choices_short = [
            str(choice).strip() for choice in raw_choices_short if str(choice).strip()
        ]
    elif isinstance(raw_choices_short, str):
        fragments = re.split(r"[\n|]", raw_choices_short)
        choices_short = [
            frag.strip(" •-\t").strip() for frag in fragments if frag.strip()
        ]

    if choices:
        while len(choices_short) < len(choices):
            fallback = choices[len(choices_short)]
            choices_short.append(str(fallback).strip())
        if len(choices_short) > len(choices):
            choices_short = choices_short[: len(choices)]
    else:
        while len(choices_short) < 3:
            choices_short.append(f"Choice {len(choices_short) + 1}")
        if len(choices_short) > 3:
            choices_short = choices_short[:3]

    normalized = dict(scene)
    normalized["scenario_display"] = scenario_display
    normalized["veo_prompt"] = veo_prompt
    normalized["sora_prompt"] = normalized["veo_prompt"]
    normalized["choices"] = choices
    normalized["choices_short"] = choices_short
    normalized["_planner_missing_prompt"] = veo_prompt_missing
    normalized["_planner_missing_prompt_reason"] = veo_prompt_missing_reason
    return normalized


def plan_initial_scene(api_key: str, base_prompt: str, model: str) -> dict:
    guidance_section = (
        f"\n\nADDITIONAL WORLD GUIDANCE:\n{PROMPT_GUIDANCE}" if PROMPT_GUIDANCE else ""
    )
    logger.info(
        "Planning initial scene with planner=%s prompt_preview=%s",
        model,
        (base_prompt[:200] + "...") if len(base_prompt) > 200 else base_prompt,
    )
    user_input = f"""
TASK: Create the opening scene with three choices.

BASE PROMPT:
{base_prompt}

Shot length: 8 seconds.
Return JSON with keys: scenario_display, sora_prompt, choices (3).
{guidance_section}
""".strip()
    raw = responses_create(
        api_key=api_key, model=model, instructions=PLANNER_SYSTEM, user_input=user_input
    )
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = "initial"
    return scene


def plan_next_scene(
    api_key: str,
    base_prompt: str,
    prior_video_prompts: List[str],
    chosen_choice: str,
    state_summaries: List[str],
    model: str,
) -> dict:
    prior_joined = (
        "\n\n---\n\n".join(prior_video_prompts)
        if prior_video_prompts
        else "(first continuation)"
    )
    state_section = (
        "\n".join(f"- {summary}" for summary in state_summaries)
        if state_summaries
        else "- No prior state summary available yet."
    )
    guidance_section = (
        f"\n\nADDITIONAL WORLD GUIDANCE:\n{PROMPT_GUIDANCE}" if PROMPT_GUIDANCE else ""
    )

    user_input = f"""
TASK: Create the next scene with three choices, continuing the story.

BASE PROMPT:
{base_prompt}

PRIOR SORA PROMPTS (in order; each was used to generate an 8s video):
{prior_joined}

CURRENT STATE SNAPSHOT (bullet list):
{state_section}

PLAYER'S CHOSEN ACTION TO CONTINUE:
{chosen_choice}

Note: The next 8-second segment MUST flow seamlessly from the prior footage, matching character positions, motion vectors, and lighting unless the chosen action forces a justified shift.

Return JSON with keys: scenario_display, sora_prompt, choices (3).
{guidance_section}
""".strip()
    raw = responses_create(
        api_key=api_key, model=model, instructions=PLANNER_SYSTEM, user_input=user_input
    )
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = "continuation"
    return scene


# === Azure Sora Helpers ===


def _api_headers(api_key: str, *, content_type: Optional[str] = None) -> Dict[str, str]:
    if not api_key:
        raise RuntimeError("Azure OpenAI API key is required")
    headers: Dict[str, str] = {"api-key": api_key}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def responses_create(
    api_key: str, model: str, instructions: str, user_input: str
) -> str:
    if not AZURE_API_BASE:
        raise RuntimeError("AZURE_OPENAI_API_BASE must be configured")

    deployment = (model or DEFAULT_PLANNER_MODEL).strip()
    if not deployment:
        raise RuntimeError("Planner model/deployment must be provided")

    logger.info(
        "Calling Azure Responses API deployment=%s instructions_len=%s input_preview=%s",
        deployment,
        len(instructions or ""),
        (user_input[:120] + "...") if len(user_input) > 120 else user_input,
    )

    candidate_versions: List[str] = []
    configured_version = (AZURE_RESPONSES_API_VERSION or "preview").strip()
    if configured_version:
        candidate_versions.append(configured_version)
    if "preview" not in [version.lower() for version in candidate_versions]:
        candidate_versions.append("preview")

    headers = _api_headers(api_key, content_type="application/json")
    payload = {"model": deployment, "instructions": instructions, "input": user_input}

    last_response: Optional[requests.Response] = None
    for version in candidate_versions:
        url = f"{AZURE_API_BASE}/openai/v1/responses?api-version={version}"
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code < 400:
            last_response = response
            break
        logger.error(
            "Responses API error status=%s version=%s body=%s",
            response.status_code,
            version,
            response.text[:800],
        )
        last_response = response

    if last_response is None:
        raise RuntimeError("Responses API request failed with no response captured")
    if last_response.status_code >= 400:
        raise RuntimeError(
            f"Responses API failed ({last_response.status_code}): {last_response.text}"
        )

    try:
        data = last_response.json()
    except ValueError as exc:
        raise RuntimeError("Responses API returned non-JSON payload") from exc

    output = data.get("output")
    if isinstance(output, str) and output.strip():
        return output
    if isinstance(output, list):
        parts: List[str] = []
        for entry in output:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                text = entry.get("text") or entry.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    content = data.get("content")
    if isinstance(content, list):
        builder: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            entries = item.get("text") or item.get("content") or item.get("parts")
            if isinstance(entries, list):
                for part in entries:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content")
                        if isinstance(text, str):
                            builder.append(text)
                    elif isinstance(part, str):
                        builder.append(part)
            elif isinstance(entries, str):
                builder.append(entries)
        if builder:
            return "\n".join(builder)

    raise RuntimeError("Responses API returned no textual output")


def _guess_mime(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0]
    return mime or "application/octet-stream"


def _video_dimensions(size: str) -> Tuple[int, int]:
    try:
        width_str, height_str = size.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    fallback_width, fallback_height = [
        int(val) for val in DEFAULT_VIDEO_SIZE.split("x")
    ]
    return fallback_width, fallback_height


def _video_jobs_url(
    *,
    job_id: Optional[str] = None,
    suffix: Optional[str] = None,
    params: Optional[Dict[str, str]] = None,
) -> str:
    if not AZURE_API_BASE:
        raise RuntimeError("AZURE_OPENAI_API_BASE must be configured")

    base = f"{AZURE_API_BASE}/openai/v1/videos"
    if job_id:
        base = f"{base}/{job_id}"
    if suffix:
        base = f"{base}/{suffix}"

    query: Dict[str, str] = {}
    if params:
        query.update(params)
    if AZURE_VIDEO_API_VERSION:
        query.setdefault("api-version", AZURE_VIDEO_API_VERSION)

    if not query:
        return base
    return f"{base}?{urlencode(query)}"


def sora_create_video(
    api_key: str,
    sora_prompt: str,
    model: str,
    size: str,
    seconds: int,
    input_reference_path: Optional[Path] = None,
) -> dict:
    width, height = _video_dimensions(size)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": sora_prompt,
        "seconds": str(seconds),
        "size": f"{width}x{height}",
    }
    if input_reference_path and input_reference_path.exists():
        try:
            with input_reference_path.open("rb") as handle:
                payload["input_video"] = {
                    "data": handle.read(),
                    "filename": input_reference_path.name,
                    "mime_type": _guess_mime(input_reference_path),
                }
        except Exception:
            logger.warning(
                "Failed to include input reference video for continuity", exc_info=True
            )

    logger.info(
        "Submitting Sora job model=%s seconds=%s size=%sx%s",
        model,
        seconds,
        width,
        height,
    )
    logger.debug(
        "Sora prompt preview: %s",
        (sora_prompt[:500] + "...") if len(sora_prompt) > 500 else sora_prompt,
    )
    response = requests.post(
        _video_jobs_url(),
        headers=_api_headers(api_key, content_type="application/json"),
        json=payload,
        timeout=600,
    )
    if response.status_code >= 400:
        logger.error(
            "Sora create failed status=%s body=%s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f"Sora create failed ({response.status_code}): {response.text}"
        )
    job = response.json()
    logger.info("Sora job submitted id=%s", job.get("id"))
    return job


def sora_retrieve_video(api_key: str, video_id: str) -> dict:
    url = _video_jobs_url(job_id=video_id)
    last_error: Optional[Exception] = None
    for attempt in range(5):
        try:
            response = requests.get(url, headers=_api_headers(api_key), timeout=120)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2**attempt, 8))
            continue

        if response.status_code >= 500 or response.status_code in (429, 520):
            last_error = RuntimeError(
                f"Sora retrieve failed ({response.status_code}): {response.text[:200]}"
            )
            time.sleep(min(2**attempt, 8))
            continue

        if response.status_code >= 400:
            raise RuntimeError(
                f"Sora retrieve failed ({response.status_code}): {response.text}"
            )

        try:
            return response.json()
        except ValueError as exc:
            last_error = exc
            time.sleep(min(2**attempt, 8))

    if last_error:
        raise RuntimeError(f"Sora retrieve failed after retries: {last_error}")
    raise RuntimeError("Sora retrieve failed after retries: unknown error")


def sora_poll_until_complete(api_key: str, job: dict) -> dict:
    video = job
    video_id = video["id"]
    pending_statuses = {
        "queued",
        "in_progress",
        "preprocessing",
        "processing",
        "running",
        "generating",
        "starting",
    }
    success_statuses = {"completed", "succeeded"}
    failure_statuses = {"failed", "cancelled", "canceled"}

    last_status = (video.get("status") or "").lower()
    last_progress = video.get("progress") or video.get("percentage")
    if last_status:
        logger.info("Polling Sora job id=%s status=%s", video_id, last_status)

    while True:
        status = (video.get("status") or "").lower()
        if status in success_statuses:
            logger.info("Sora job id=%s reached terminal status=%s", video_id, status)
            return video
        if status in failure_statuses:
            break
        if status and status not in pending_statuses:
            logger.warning(
                "Sora job id=%s encountered unexpected status=%s; continuing to poll",
                video_id,
                status,
            )

        time.sleep(2)
        video = sora_retrieve_video(api_key, video_id)
        status = (video.get("status") or "").lower()
        progress = video.get("progress") or video.get("percentage")
        if status != last_status or progress != last_progress:
            logger.info(
                "Polling Sora job id=%s status=%s progress=%s",
                video_id,
                status,
                progress,
            )
            last_status = status
            last_progress = progress

    final_status = (video.get("status") or "").lower()
    error_payload = video.get("error")
    failure_reason = video.get("failure_reason")
    logger.error(
        "Sora job failed id=%s status=%s error=%s failure_reason=%s",
        video_id,
        final_status or video.get("status"),
        error_payload,
        failure_reason,
    )
    detail_parts = [f"Job {video_id} failed"]
    if failure_reason:
        detail_parts.append(f"reason={failure_reason}")
    if isinstance(error_payload, dict):
        message = error_payload.get("message")
        code = error_payload.get("code") or error_payload.get("type")
        inner = error_payload.get("innererror") or error_payload.get("inner_error")
        if code:
            detail_parts.append(f"code={code}")
        if message:
            detail_parts.append(f"message={message}")
        if inner:
            detail_parts.append(f"details={inner}")
    elif isinstance(error_payload, str):
        detail_parts.append(error_payload)
    else:
        detail_parts.append(str(video))
    raise RuntimeError("; ".join(detail_parts))


def _iter_dicts(obj: Any) -> List[Dict[str, Any]]:
    stack = [obj]
    collected: List[dict] = []
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            collected.append(current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return collected


def _pick_download_url(video: dict, variant: str) -> Optional[str]:
    variant_lower = (variant or "").lower()
    candidates: List[Tuple[int, str]] = []
    for entry in _iter_dicts(video):
        url = None
        for key in (
            "download_url",
            "downloadUrl",
            "content_url",
            "contentUrl",
            "asset_url",
            "assetUrl",
        ):
            value = entry.get(key)
            if isinstance(value, str) and value.startswith("http"):
                url = value
                break
        if not url and "url" in entry:
            value = entry.get("url")
            if (
                isinstance(value, str)
                and value.startswith("http")
                and any(
                    entry.get(meta_key)
                    for meta_key in (
                        "variant",
                        "asset_type",
                        "assetType",
                        "media_type",
                        "mime_type",
                        "content_type",
                        "file_name",
                        "filename",
                    )
                )
            ):
                url = value
        if not url:
            continue

        asset_labels = " ".join(
            str(entry.get(key, ""))
            for key in (
                "variant",
                "asset_type",
                "assetType",
                "type",
                "purpose",
                "role",
                "label",
            )
            if entry.get(key) is not None
        ).lower()
        media_type = str(
            entry.get("media_type")
            or entry.get("mime_type")
            or entry.get("content_type")
            or ""
        ).lower()
        filename = str(entry.get("file_name") or entry.get("filename") or "").lower()

        priority = 5
        if variant_lower:
            if variant_lower == "video":
                if (
                    "video" in asset_labels
                    or "video" in media_type
                    or filename.endswith((".mp4", ".mov", ".webm", ".mkv"))
                ):
                    priority = 0
            elif (
                variant_lower in asset_labels
                or variant_lower in media_type
                or variant_lower in filename
            ):
                priority = 1
        else:
            priority = 2

        if priority == 5 and ("video" in asset_labels or "video" in media_type):
            priority = 1

        candidates.append((priority, url))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _stream_download(url: str, out_path: Path) -> None:
    with requests.get(url, stream=True, timeout=1800) as response:
        if response.status_code >= 400:
            raise RuntimeError(
                f"Sora asset download failed ({response.status_code}): {response.text}"
            )
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)


def sora_download_content(
    api_key: str, video: dict, out_path: Path, variant: str = "video"
) -> Path:
    video_id = video.get("id")
    if not isinstance(video_id, str):
        raise RuntimeError("Sora download failed: video job missing id")

    direct_url = _pick_download_url(video, variant)
    if direct_url:
        logger.info(
            "Downloading Sora asset via direct URL id=%s variant=%s", video_id, variant
        )
        logger.debug("Direct download URL: %s", direct_url)
        _stream_download(direct_url, out_path)
        return out_path

    logger.info(
        "Falling back to Sora content endpoint id=%s variant=%s", video_id, variant
    )
    url = _video_jobs_url(
        job_id=video_id, suffix="content", params={"variant": variant}
    )
    with requests.get(
        url,
        headers=_api_headers(api_key),
        stream=True,
        timeout=1800,
    ) as response:
        if response.status_code >= 400:
            raise RuntimeError(
                f"Sora download failed ({response.status_code}): {response.text}"
            )
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return out_path


def generate_scene_video(
    api_key: str,
    sora_prompt: str,
    model: str,
    size: str,
    seconds: int,
    input_reference: Optional[Path],
) -> Tuple[str, Path, Path]:
    seconds = normalize_seconds(seconds)
    logger.info("Awaiting Sora job slot current_limit=%s", MAX_CONCURRENT_VIDEO_JOBS)
    VIDEO_JOB_SEMAPHORE.acquire()
    logger.info("Sora job slot acquired")
    try:
        job = sora_create_video(
            api_key=api_key,
            sora_prompt=sora_prompt,
            model=model,
            size=size,
            seconds=seconds,
            input_reference_path=input_reference,
        )

        video = sora_poll_until_complete(api_key, job)
        video_id = video["id"]

        video_path = VIDEO_DIR / f"{video_id}.mp4"
        sora_download_content(api_key, video, video_path, variant="video")

        last_frame_path = FRAME_DIR / f"{video_id}_last.jpg"
        extract_last_frame(video_path, last_frame_path)
        logger.info(
            "Sora job complete id=%s video_path=%s frame_path=%s",
            video_id,
            video_path,
            last_frame_path,
        )
        return video_id, video_path, last_frame_path
    finally:
        VIDEO_JOB_SEMAPHORE.release()
        logger.info("Sora job slot released")


def _video_duration_seconds(path: Path) -> Optional[float]:
    try:
        if cv2 is not None:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                cap.release()
                if fps > 0 and frame_count > 0:
                    return frame_count / fps
    except Exception:
        logger.debug("[veo] cv2 duration probe failed", exc_info=True)

    if FFMPEG_BIN:
        try:
            cmd = [
                FFMPEG_BIN,
                "-i",
                str(path),
                "-hide_banner",
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            output = proc.stderr or proc.stdout
            if output:
                for line in output.splitlines():
                    if "Duration:" in line:
                        duration_token = (
                            line.split("Duration:", 1)[1].split(",", 1)[0].strip()
                        )
                        h, m, s = duration_token.split(":")
                        return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            logger.debug("[veo] ffmpeg duration probe failed", exc_info=True)
    return None


def extract_last_frame(video_path: Path, out_image_path: Path) -> Path:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            success, frame = False, None
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
                success, frame = cap.read()
            if not success or frame is None:
                cap.release()
                cap = cv2.VideoCapture(str(video_path))
                while True:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    frame = fr
                    success = True
            cap.release()
            if success and frame is not None:
                if cv2.imwrite(str(out_image_path), frame):
                    return out_image_path

    if FFMPEG_BIN:
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-sseof",
            "-0.05",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(out_image_path),
        ]
        subprocess.check_call(cmd)
        if out_image_path.exists():
            return out_image_path

    raise RuntimeError(
        "Failed to extract last frame: OpenCV/FFmpeg unavailable or video unreadable."
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
