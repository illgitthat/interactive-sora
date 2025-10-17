import json
import logging
import mimetypes
import re
import os
import subprocess
import threading
import time
from urllib.parse import urlencode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import imageio_ffmpeg  # type: ignore

    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover - optional dependency
    FFMPEG_BIN = None

APP_TITLE = "Sora Choose-Your-Own Adventure API"

VIDEO_DIR = Path("sora_cyoa_videos")
FRAME_DIR = Path("sora_cyoa_frames")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VIDEO_SIZE = "1280x720"
DEFAULT_SECONDS = 8
ALLOWED_SECONDS = [4, 8, 12]

AZURE_API_BASE = os.getenv("AZURE_OPENAI_API_BASE", "").rstrip("/")
AZURE_RESPONSES_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "preview").strip()
AZURE_VIDEO_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", AZURE_RESPONSES_API_VERSION or "preview"
).strip()
DEFAULT_PLANNER_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-5-chat").strip() or "gpt-5-chat"
)
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
DEFAULT_SORA_MODEL = os.getenv("AZURE_OPENAI_SORA_MODEL", "sora-2").strip() or "sora-2"

LOG_LEVEL_NAME = os.getenv("SORA_LOG_LEVEL", "INFO").strip().upper() or "INFO"
logger = logging.getLogger("interactive_sora")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL_NAME, logging.INFO))

PLANNER_SYSTEM = """
You are the Scenario Planner for a Sora-powered choose-your-own-adventure game.

Your job:
- Given a BASE PROMPT (world/tone) or a CONTINUATION (previous scene prompts + the player's chosen action),
- Produce a JSON object that contains:
  {
    "scenario_display": "A short paragraph (<= 120 words) narrating the current scene to show in the UI.",
    "sora_prompt": "<A detailed Sora prompt for generating an 8-second video.>",
    "choices": ["<choice 1>", "<choice 2>", "<choice 3>"]
  }

Rules:
1) The 'sora_prompt' must be the exact text we send to Sora's /videos API.
   - Include a line: "Context (not visible in video, only for AI guidance): ..." to carry forward continuity and constraints.
   - Include a line: "Prompt: ..." with concrete, cinematic directions (camera, subject, motion, lighting).
   - Keep 'Prompt' specific to a single 8-second shot.
   - For steps after the first, begin exactly from the final frame of the previous scene.

2) Safety & platform constraints (strict):
   - Content must be suitable for audiences under 18.
   - Do NOT depict real people (including public figures) or copyrighted/fictional characters.
   - Avoid copyrighted music and explicit logos/trademarks. Use generic brand cues only.
   - Avoid hate, sexual content, excessive violence, or self-harm.

3) Continuity:
   - Maintain consistent characters, setting, tone, camera language, and lighting unless the choice implies a justified shift.
   - Ensure smooth shot-to-shot transitions (same time of day, matching positions/poses as appropriate).

4) Choices
   - Provide exactly three distinct options for what the player can do next.
   - Make each option feasible in the next short shot, and clearly different in intent.
   - Keep each choice concise (<= 22 words).

5) Output strictly JSON. No markdown, no commentary, no code fences.
""".strip()


class CreateSessionRequest(BaseModel):
    api_key: str = Field(..., alias="apiKey")
    planner_model: str = Field(..., alias="plannerModel")
    sora_model: str = Field(..., alias="soraModel")
    video_size: str = Field(DEFAULT_VIDEO_SIZE, alias="videoSize")
    base_prompt: str = Field(..., alias="basePrompt")
    max_steps: int = Field(10, alias="maxSteps")

    @validator("max_steps")
    def validate_max_steps(cls, value: int) -> int:
        if not 1 <= value <= 30:
            raise ValueError("maxSteps must be between 1 and 30")
        return value


class ChoiceRequest(BaseModel):
    choice_index: int = Field(..., alias="choiceIndex")

    @validator("choice_index")
    def validate_choice_index(cls, value: int) -> int:
        if value not in (0, 1, 2):
            raise ValueError("choiceIndex must be 0, 1, or 2")
        return value


class StoryStepResponse(BaseModel):
    scene_number: int = Field(..., alias="sceneNumber")
    scenario_display: str = Field(..., alias="scenarioDisplay")
    sora_prompt: str = Field(..., alias="soraPrompt")
    choices: List[str]
    choice_index: Optional[int] = Field(None, alias="choiceIndex")
    video_url: Optional[str] = Field(None, alias="videoUrl")
    poster_url: Optional[str] = Field(None, alias="posterUrl")
    video_id: Optional[str] = Field(None, alias="videoId")
    planner_missing_prompt: bool = Field(False, alias="plannerMissingPrompt")
    planner_missing_prompt_reason: str = Field("", alias="plannerMissingPromptReason")


class SessionResponse(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    story: List[StoryStepResponse]
    step_count: int = Field(..., alias="stepCount")
    max_steps: int = Field(..., alias="maxSteps")
    has_remaining_steps: bool = Field(..., alias="hasRemainingSteps")


@dataclass
class SessionConfig:
    api_key: str
    planner_model: str
    sora_model: str
    video_size: str
    base_prompt: str
    max_steps: int


@dataclass
class SessionState:
    config: SessionConfig
    story: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


SESSIONS: Dict[str, SessionState] = {}

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
app.mount("/media/frames", StaticFiles(directory=str(FRAME_DIR)), name="frames")


def _api_headers(api_key: str, *, content_type: Optional[str] = None) -> Dict[str, str]:
    if not api_key:
        raise RuntimeError("Azure OpenAI API key is required")
    headers: Dict[str, str] = {
        "api-key": api_key,
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def responses_create(
    api_key: str, model: str, instructions: str, user_input: str
) -> str:
    if not AZURE_API_BASE:
        raise RuntimeError("AZURE_OPENAI_API_BASE must be configured")

    deployment = (model or DEFAULT_PLANNER_DEPLOYMENT).strip()
    if not deployment:
        raise RuntimeError("Planner model/deployment must be provided")

    logger.info(
        "Calling Azure Responses API deployment=%s instructions_len=%s input_preview=%s",
        deployment,
        len(instructions or ""),
        (user_input[:120] + "...") if len(user_input) > 120 else user_input,
    )

    candidate_versions = []
    configured_version = (AZURE_RESPONSES_API_VERSION or "preview").strip()
    if configured_version:
        candidate_versions.append(configured_version)
    if "preview" not in [version.lower() for version in candidate_versions]:
        candidate_versions.append("preview")

    headers = _api_headers(api_key, content_type="application/json")
    payload = {
        "model": deployment,
        "instructions": instructions,
        "input": user_input,
    }

    last_error: Optional[Exception] = None
    last_response: Optional[requests.Response] = None
    for version in candidate_versions:
        url = f"{AZURE_API_BASE}/openai/v1/responses?api-version={version}"
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code < 400:
            last_response = response
            break

        error_text = response.text
        logger.error(
            "Responses API error status=%s version=%s body=%s",
            response.status_code,
            version,
            error_text,
        )
        if (
            response.status_code == 400
            and "api version not supported" in error_text.lower()
            and version.lower() != "preview"
        ):
            last_error = RuntimeError(
                f"Responses API error {response.status_code}: {error_text}"
            )
            continue

        raise RuntimeError(f"Responses API error {response.status_code}: {error_text}")

    if last_response is None:
        if last_error:
            raise last_error
        raise RuntimeError("Responses API call failed without a successful attempt")

    data = last_response.json()

    text = data.get("output_text", "")
    if text:
        return text

    try:
        items = data.get("output", [])
        builder: List[str] = []
        for item in items:
            blocks = item.get("content") or []
            for block in blocks:
                b_type = block.get("type")
                if b_type in {"output_text", "text"}:
                    builder.append(block.get("text", ""))
                elif isinstance(block.get("text"), list):
                    for segment in block["text"]:
                        if isinstance(segment, dict) and segment.get("type") in {
                            "output_text",
                            "text",
                        }:
                            builder.append(segment.get("text", ""))
        if builder:
            return "".join(builder)
    except Exception:
        pass

    return json.dumps(data)


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
    sora_prompt_keys = [
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

    scenario_display = _pick(scenario_display_keys)
    if isinstance(scenario_display, list):
        parts = [str(item).strip() for item in scenario_display if str(item).strip()]
        scenario_display = " ".join(parts)
    if not scenario_display:
        scenario_display = (
            "Planner response missing scene description. Adjust your prompt and retry."
        )

    sora_prompt_raw = _pick(sora_prompt_keys)
    sora_prompt_missing = False
    sora_prompt_missing_reason = ""

    sora_prompt_value: Any = sora_prompt_raw
    if isinstance(sora_prompt_value, dict):
        lines: List[str] = []
        for key, value in sora_prompt_value.items():
            if value is None:
                continue
            text_val = str(value).strip()
            if not text_val:
                continue
            lines.append(f"{key}: {text_val}")
        if lines:
            sora_prompt_value = "\n".join(lines).strip()
        else:
            sora_prompt_missing = True
            sora_prompt_missing_reason = (
                "Planner returned prompt dict but it had no usable values."
            )
            sora_prompt_value = ""
    elif isinstance(sora_prompt_value, list):
        joined = "\n".join(
            str(item).strip() for item in sora_prompt_value if str(item).strip()
        )
        if joined:
            sora_prompt_value = joined
        else:
            sora_prompt_missing = True
            sora_prompt_missing_reason = (
                "Planner returned prompt list but all entries were empty."
            )
            sora_prompt_value = ""

    if sora_prompt_value is None:
        sora_prompt_missing = True
        if not sora_prompt_missing_reason:
            sora_prompt_missing_reason = (
                "Planner response missing recognized Sora prompt field."
            )
        sora_prompt_value = ""
    elif not isinstance(sora_prompt_value, str):
        sora_prompt_value = str(sora_prompt_value).strip()
        if not sora_prompt_value:
            sora_prompt_missing = True
            if not sora_prompt_missing_reason:
                sora_prompt_missing_reason = (
                    "Planner returned non-string prompt that was empty after casting."
                )
    else:
        sora_prompt_value = sora_prompt_value.strip()
        if not sora_prompt_value:
            sora_prompt_missing = True
            if not sora_prompt_missing_reason:
                sora_prompt_missing_reason = "Planner Sora prompt string was blank."

    sora_prompt = (
        "Planner response missing Sora prompt details. Please tweak your base prompt or retry."
        if sora_prompt_missing
        else sora_prompt_value
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

    normalized = dict(scene)
    normalized["scenario_display"] = scenario_display
    normalized["sora_prompt"] = sora_prompt
    normalized["choices"] = choices
    normalized["_planner_missing_prompt"] = sora_prompt_missing
    normalized["_planner_missing_prompt_reason"] = sora_prompt_missing_reason
    return normalized


def plan_initial_scene(api_key: str, base_prompt: str, model: str) -> dict:
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
""".strip()
    raw = responses_create(
        api_key=api_key, model=model, instructions=PLANNER_SYSTEM, user_input=user_input
    )
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = "initial"
    logger.info(
        "Planner initial scene ready missing_prompt=%s choices=%s",
        scene.get("_planner_missing_prompt", False),
        scene.get("choices", []),
    )
    return scene


def plan_next_scene(
    api_key: str,
    base_prompt: str,
    prior_sora_prompts: List[str],
    chosen_choice: str,
    model: str,
) -> dict:
    prior_joined = "\n\n---\n\n".join(prior_sora_prompts)
    logger.info(
        "Planning next scene with planner=%s choice=%s prior_prompt_count=%s",
        model,
        chosen_choice,
        len(prior_sora_prompts),
    )
    user_input = f"""
TASK: Create the next scene with three choices, continuing the story.

BASE PROMPT:
{base_prompt}

PRIOR SORA PROMPTS (in order; each was used to generate an 8s video):
{prior_joined}

PLAYER'S CHOSEN ACTION TO CONTINUE:
{chosen_choice}

Note: The next 8-second shot MUST begin exactly from the final frame of the previous shot,
preserving continuity (subjects, camera position, lighting, motion direction), unless the chosen action implies a change.

Return JSON with keys: scenario_display, sora_prompt, choices (3).
""".strip()
    raw = responses_create(
        api_key=api_key, model=model, instructions=PLANNER_SYSTEM, user_input=user_input
    )
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = "continuation"
    logger.info(
        "Planner continuation scene ready missing_prompt=%s choices=%s",
        scene.get("_planner_missing_prompt", False),
        scene.get("choices", []),
    )
    return scene


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

    base = f"{AZURE_API_BASE}/openai/v1/video/generations/jobs"
    if job_id:
        base = f"{base}/{job_id}"
    if suffix:
        base = f"{base}/{suffix}"

    query: Dict[str, str] = {"api-version": AZURE_VIDEO_API_VERSION or "preview"}
    if params:
        query.update(params)
    return f"{base}?{urlencode(query)}"


def _video_generation_content_url(
    generation_id: str, variant: Optional[str] = None
) -> str:
    if not AZURE_API_BASE:
        raise RuntimeError("AZURE_OPENAI_API_BASE must be configured")

    base = f"{AZURE_API_BASE}/openai/v1/video/generations/{generation_id}/content"
    if variant:
        base = f"{base}/{variant}"

    query: Dict[str, str] = {"api-version": AZURE_VIDEO_API_VERSION or "preview"}
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
        "prompt": sora_prompt,
        "model": model,
        "n_variants": 1,
        "n_seconds": seconds,
        "width": width,
        "height": height,
    }

    # Azure video preview currently expects JSON payloads; input reference continuity is handled in prompt context.

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
    start_time = time.time()
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
    logger.info(
        "Sora job submitted id=%s latency=%.2fs",
        job.get("id"),
        time.time() - start_time,
    )
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


def _iter_dicts(obj: Any) -> List[Dict[str, Any]]:
    stack = [obj]
    dicts: List[dict] = []
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            dicts.append(current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return dicts


def _collect_generation_ids(video: dict, *, job_id: Optional[str] = None) -> List[str]:
    candidates: List[str] = []

    def _append_if_valid(candidate: Optional[str]) -> None:
        if not isinstance(candidate, str):
            return
        if candidate and candidate != job_id and candidate not in candidates:
            candidates.append(candidate)

    generations = video.get("generations")
    if isinstance(generations, list):
        for entry in generations:
            if isinstance(entry, dict):
                _append_if_valid(entry.get("id"))

    output = video.get("output")
    if isinstance(output, dict):
        data = output.get("data")
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    _append_if_valid(entry.get("id"))

    for entry in _iter_dicts(video):
        if isinstance(entry, dict):
            _append_if_valid(entry.get("generation_id"))
            # Some payloads surface generation identifiers under "asset" records.
            if entry.get("object") in {"video.generation", "video.asset"}:
                _append_if_valid(entry.get("id"))

    return candidates


def _pick_download_url(video: dict, variant: str) -> Optional[str]:
    # Scan the job payload for any downloadable asset URLs and prefer the requested variant.
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
        if not url:
            # Accept bare 'url' only if accompanied by asset metadata to avoid noise.
            if "url" in entry:
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
        with open(out_path, "wb") as handle:
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
        _stream_download(direct_url, out_path)
        return out_path

    generation_ids = _collect_generation_ids(video, job_id=video_id)
    for generation_id in generation_ids:
        try:
            logger.info(
                "Downloading Sora asset via generation id=%s variant=%s",
                generation_id,
                variant,
            )
            url = _video_generation_content_url(generation_id, variant)
            with requests.get(
                url,
                headers=_api_headers(api_key),
                stream=True,
                timeout=1800,
            ) as response:
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Sora generation download failed ({response.status_code}): {response.text}"
                    )
                with open(out_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
            return out_path
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.warning(
                "Sora generation download attempt failed job_id=%s generation_id=%s error=%s",
                video_id,
                generation_id,
                exc,
            )

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
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path


def sora_poll_until_complete(api_key: str, job: dict) -> dict:
    video = job
    video_id = video["id"]
    pending_statuses = {
        "queued",
        "in_progress",
        "preprocessing",
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


def normalize_seconds(secs: int) -> int:
    return min(ALLOWED_SECONDS, key=lambda value: abs(value - int(secs)))


def generate_scene_video(
    api_key: str,
    sora_prompt: str,
    model: str,
    size: str,
    seconds: int,
    input_reference: Optional[Path],
) -> Tuple[str, Path, Path]:
    seconds = normalize_seconds(seconds)
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


def create_story_item(scene: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_display": scene["scenario_display"],
        "sora_prompt": scene["sora_prompt"],
        "choices": scene["choices"],
        "choice_index": None,
        "video_id": None,
        "video_path": None,
        "last_frame_path": None,
        "planner_missing_prompt": scene.get("_planner_missing_prompt", False),
        "planner_missing_prompt_reason": scene.get(
            "_planner_missing_prompt_reason", ""
        ),
        "planner_raw_output": scene.get("_raw_planner_output", ""),
        "planner_model": scene.get("_planner_model", ""),
        "planner_stage": scene.get("_planner_stage", ""),
    }


def build_story_payload(story: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, step in enumerate(story, start=1):
        video_url = None
        if step.get("video_path"):
            video_url = f"/media/videos/{Path(step['video_path']).name}"

        poster_url = None
        if step.get("last_frame_path"):
            poster_url = f"/media/frames/{Path(step['last_frame_path']).name}"

        payload.append(
            {
                "sceneNumber": idx,
                "scenarioDisplay": step.get("scenario_display", ""),
                "soraPrompt": step.get("sora_prompt", ""),
                "choices": step.get("choices", []),
                "choiceIndex": step.get("choice_index"),
                "videoUrl": video_url,
                "posterUrl": poster_url,
                "videoId": step.get("video_id"),
                "plannerMissingPrompt": step.get("planner_missing_prompt", False),
                "plannerMissingPromptReason": step.get(
                    "planner_missing_prompt_reason", ""
                ),
            }
        )
    return payload


def serialize_session(session_id: str, state: SessionState) -> SessionResponse:
    story_payload = build_story_payload(state.story)
    response = SessionResponse(
        sessionId=session_id,
        story=[StoryStepResponse(**item) for item in story_payload],
        stepCount=state.step_count,
        maxSteps=state.config.max_steps,
        hasRemainingSteps=state.step_count < state.config.max_steps,
    )
    return response


def get_session_or_404(session_id: str) -> SessionState:
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/default-config")
def default_config() -> Dict[str, Any]:
    return {
        "apiKey": DEFAULT_API_KEY,
        "plannerModel": DEFAULT_PLANNER_DEPLOYMENT,
        "soraModel": DEFAULT_SORA_MODEL,
        "videoSize": DEFAULT_VIDEO_SIZE,
        "maxSteps": 10,
    }


@app.post("/api/session", response_model=SessionResponse)
def create_session(payload: CreateSessionRequest) -> SessionResponse:
    config = SessionConfig(
        api_key=(payload.api_key or DEFAULT_API_KEY).strip(),
        planner_model=(payload.planner_model or DEFAULT_PLANNER_DEPLOYMENT).strip()
        or DEFAULT_PLANNER_DEPLOYMENT,
        sora_model=(payload.sora_model or DEFAULT_SORA_MODEL).strip()
        or DEFAULT_SORA_MODEL,
        video_size=payload.video_size.strip() or DEFAULT_VIDEO_SIZE,
        base_prompt=payload.base_prompt.strip(),
        max_steps=payload.max_steps,
    )

    if not config.api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    if not config.base_prompt:
        raise HTTPException(status_code=400, detail="Base prompt is required")

    session_id = uuid4().hex
    logger.info(
        "Creating session id=%s planner=%s sora=%s size=%s max_steps=%s",
        session_id,
        config.planner_model,
        config.sora_model,
        config.video_size,
        config.max_steps,
    )
    state = SessionState(config=config)
    SESSIONS[session_id] = state

    try:
        scene = plan_initial_scene(
            api_key=config.api_key,
            base_prompt=config.base_prompt,
            model=config.planner_model,
        )
        story_item = create_story_item(scene)

        if not story_item["planner_missing_prompt"]:
            logger.info("Session %s requesting initial video", session_id)
            try:
                video_id, video_path, last_frame_path = generate_scene_video(
                    api_key=config.api_key,
                    sora_prompt=story_item["sora_prompt"],
                    model=config.sora_model,
                    size=config.video_size,
                    seconds=DEFAULT_SECONDS,
                    input_reference=None,
                )
            except Exception:
                logger.exception("Session %s failed during initial video", session_id)
                raise
            story_item["video_id"] = video_id
            story_item["video_path"] = str(video_path)
            story_item["last_frame_path"] = str(last_frame_path)
        else:
            story_item["planner_missing_prompt"] = True

        state.story.append(story_item)
    except Exception as exc:
        SESSIONS.pop(session_id, None)
        logger.exception("Session %s initialization failed: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return serialize_session(session_id, state)


@app.get("/api/session")
def list_sessions() -> Dict[str, Any]:
    return {
        "sessions": [
            {
                "sessionId": session_id,
                "stepCount": state.step_count,
                "maxSteps": state.config.max_steps,
            }
            for session_id, state in SESSIONS.items()
        ]
    }


@app.get("/api/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    state = get_session_or_404(session_id)
    return serialize_session(session_id, state)


@app.post("/api/session/{session_id}/choice", response_model=SessionResponse)
def advance_story(session_id: str, payload: ChoiceRequest) -> SessionResponse:
    state = get_session_or_404(session_id)
    with state.lock:
        if not state.story:
            raise HTTPException(
                status_code=400, detail="Session has not been initialized"
            )

        current = state.story[-1]
        if current.get("choice_index") is not None:
            raise HTTPException(
                status_code=400, detail="Current scene already has a recorded choice"
            )

        current["choice_index"] = payload.choice_index
        chosen_choice = current["choices"][payload.choice_index]
        state.step_count += 1

        logger.info(
            "Session %s advancing choice_index=%s choice=%s step=%s/%s",
            session_id,
            payload.choice_index,
            chosen_choice,
            state.step_count,
            state.config.max_steps,
        )

        if state.step_count >= state.config.max_steps:
            return serialize_session(session_id, state)

        prior_sora_prompts = [step["sora_prompt"] for step in state.story]

        try:
            next_scene = plan_next_scene(
                api_key=state.config.api_key,
                base_prompt=state.config.base_prompt,
                prior_sora_prompts=prior_sora_prompts,
                chosen_choice=chosen_choice,
                model=state.config.planner_model,
            )
            next_item = create_story_item(next_scene)

            input_reference = None
            if current.get("last_frame_path"):
                input_reference = Path(current["last_frame_path"])

            if not next_item["planner_missing_prompt"]:
                logger.info("Session %s requesting continuation video", session_id)
                try:
                    video_id, video_path, last_frame_path = generate_scene_video(
                        api_key=state.config.api_key,
                        sora_prompt=next_item["sora_prompt"],
                        model=state.config.sora_model,
                        size=state.config.video_size,
                        seconds=DEFAULT_SECONDS,
                        input_reference=input_reference,
                    )
                except Exception:
                    logger.exception(
                        "Session %s failed during continuation video", session_id
                    )
                    raise
                next_item["video_id"] = video_id
                next_item["video_path"] = str(video_path)
                next_item["last_frame_path"] = str(last_frame_path)
            else:
                next_item["planner_missing_prompt"] = True

            state.story.append(next_item)
        except Exception as exc:
            logger.exception("Session %s advance failed: %s", session_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return serialize_session(session_id, state)


if __name__ == "__main__":  # pragma: no cover - convenience for local dev
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
