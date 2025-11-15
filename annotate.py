import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import cv2

from ollama import OllamaClient

_schema_cache: Optional[dict] = None

def _minify_system_text(text: str) -> str:
    """
    Trim comments and blanks to reduce prompt size.
    - Drops lines starting with '#' or '---' separators.
    - Collapses extra whitespace.
    """
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or set(s) <= {"-", "—"}:
            continue
        kept.append(s)
    return "\n".join(kept)

def load_annotation_schema(project_root: str) -> dict:
    """
    Load and cache annotation.schema.json.
    """
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    candidates = [
        os.path.join(os.path.dirname(__file__), "schema", "annotation.schema.json")
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _schema_cache = json.load(f)
            # Print only the file name
            print(f"Using annotation schema: {os.path.basename(path)}")
            return _schema_cache

    tried = "\n  - ".join(os.path.abspath(p) for p in candidates)
    raise FileNotFoundError(f"Schema file not found. Tried:\n  - {tried}")

def load_system_prompt(project_root: str, image_count: int, film: Dict) -> str:
    """
    Load system prompt from the local repository's system.txt only.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "system.txt")
    if not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    prompt = prompt.replace("{title}", film.get('title') or film.get('Title', 'Unknown'))
    prompt = prompt.replace("{year}", str(film.get('year', 'Unknown')))
    prompt = prompt.replace("{director}", film.get('director', 'Unknown'))
    prompt = prompt.replace("{image-count}", str(image_count))
    return prompt

def has_scenes(shotlist: List[Dict]) -> bool:
    for shot in shotlist:
        if (shot.get('Scene') or '').strip():
            return True
    return False

def get_unique_scenes(shotlist: List[Dict]) -> List[str]:
    scenes = set()
    for shot in shotlist:
        s = (shot.get('Scene') or '').strip()
        if s:
            scenes.add(s)
    return sorted(list(scenes), key=lambda x: int(x) if x.isdigit() else 0)

def parse_timecode(tc: str) -> float:
    parts = (tc or "").split(':')
    if len(parts) == 3:
        hh, mm, ss = parts
    elif len(parts) == 2:
        hh, (mm, ss) = 0, parts
    else:
        return 0.0
    try:
        hh = int(hh)
        mm = int(mm)
        ss = float(ss)
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return 0.0

def extract_frame_at_time(video_path: str, timestamp: float, output_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_index = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return cv2.imwrite(output_path, frame)

def extract_frames_for_shot(video_path: str, start_tc: str, end_tc: str, output_dir: str, movie_base: str, shot_index: int) -> List[str]:
    t0 = parse_timecode(start_tc)
    t1 = parse_timecode(end_tc)
    if t1 <= t0:
        return []
    duration = max(0.01, t1 - t0)
    samples = 4
    times = [t0 + duration * (i + 1) / (samples + 1) for i in range(samples)]
    out_paths: List[str] = []
    shot_dir = os.path.join(output_dir, movie_base, f"shot_{shot_index:04d}")
    for idx, ts in enumerate(times, start=1):
        out = os.path.join(shot_dir, f"frame_{idx:02d}.jpg")
        if extract_frame_at_time(video_path, ts, out):
            out_paths.append(out)
    return out_paths

def _ensure_list_fields(obj: dict) -> dict:
    out = {"Protagonists": [], "Place": [], "Actions": [], "Objects": []}
    for k in out.keys():
        v = obj.get(k, [])
        if v is None:
            v = []
        if isinstance(v, str):
            v = [v] if v.strip() else []
        elif isinstance(v, (int, float, bool)):
            v = [str(v)]
        elif isinstance(v, list):
            v = [str(x) for x in v if x is not None and str(x).strip()]
        else:
            v = []
        out[k] = v
    return out

def _minify(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)

def annotate_shot(
    shot: Dict,
    index: int,
    video_path: str,
    film: Dict,
    ollama: OllamaClient,
    frames_dir: str,
    project_root: str,
    verbose: bool = False
) -> tuple[str, float]:
    t0 = time.perf_counter()

    # Skip ignored or invalid rows
    if (shot.get('Ignore') or '').strip().lower() == 'yes':
        return "", 0.0
    start_tc = shot.get('Start') or shot.get('TC In') or ''
    end_tc = shot.get('End') or shot.get('TC Out') or ''
    if not start_tc or not end_tc:
        return "", 0.0

    # Extract frames
    movie_filename = (film.get('filename') or film.get('Filename') or '')
    movie_base = os.path.splitext(movie_filename)[0]
    t_extract0 = time.perf_counter()
    image_paths = extract_frames_for_shot(video_path, start_tc, end_tc, frames_dir, movie_base, index-1)
    extract_s = time.perf_counter() - t_extract0
    if not image_paths:
        dur = time.perf_counter() - t0
        if verbose:
            print(f"    details: frames=0 | extract {extract_s:.2f}s | model 0.00s | parse 0.00s")
        return "", dur

    system_text = load_system_prompt(project_root, len(image_paths), film)
    user_prompt = 'Return a concise JSON object: {"caption": "<one sentence visual description>"}'

    # Model call
    t_model0 = time.perf_counter()
    try:
        if hasattr(ollama, "generate_with_images"):
            resp = ollama.generate_with_images(prompt=user_prompt, image_paths=image_paths, system=system_text, stream=False)
        else:
            resp = ollama.generate(prompt=user_prompt, system=system_text, stream=False)  # fallback
    except Exception as e:
        print(f"[warn] Ollama call failed for shot {index}: {e}")
        resp = None
    model_s = time.perf_counter() - t_model0

    # Parse response
    caption = ""
    t_parse0 = time.perf_counter()
    if resp:
        try:
            data = json.loads(resp)
            caption = data.get("caption") or resp.strip()
        except Exception:
            caption = resp.strip()
    parse_s = time.perf_counter() - t_parse0 if resp else 0.0

    dur = time.perf_counter() - t0
    if verbose:
        print(f"    details: frames={len(image_paths)} | extract {extract_s:.2f}s | model {model_s:.2f}s | parse {parse_s:.2f}s")
    return caption, dur

def annotate_shots(
    shotlist: List[Dict],
    video_path: str,
    film: Dict,
    ollama: OllamaClient,
    frames_dir: str,
    project_root: str,
    limit: Optional[int] = None,
    start_index: int = 1,
    verbose: bool = False,
    save_callback=None  # New: called after each shot to save progress
) -> List[Dict]:
    os.makedirs(frames_dir, exist_ok=True)
    total = len(shotlist)
    plan = (min(limit, max(0, total - (start_index - 1))) if limit is not None else max(0, total - (start_index - 1)))
    planned_end = min(total, max(start_index, 1) + max(plan, 0) - 1)

    print(f"Starting annotation: {plan} planned shots (from {start_index} to {planned_end} of {total})")

    processed = 0
    sum_dur = 0.0
    t_total0 = time.perf_counter()

    for i, shot in enumerate(shotlist, start=1):
        if i < max(1, start_index):
            continue
        if limit is not None and processed >= limit:
            break
        if (shot.get('Ignore') or '').strip().lower() == 'yes':
            shot['Shot_Caption'] = ""
            continue

        print(f"Processing shot {processed+1}/{plan} (index {i} of {total})...")
        caption, dur = annotate_shot(shot, i, video_path, film, ollama, frames_dir, project_root, verbose=verbose)
        shot['Shot_Caption'] = caption

        # Save progress after each shot
        if save_callback:
            save_callback(shotlist)

        processed += 1
        sum_dur += dur
        avg = sum_dur / processed if processed else 0.0
        remaining = max(0, plan - processed)
        eta = remaining * avg
        mm, ss = divmod(int(eta), 60)
        pct = (processed / plan * 100.0) if plan else 100.0

        print(f"  ✓ {dur:.2f}s | progress {processed}/{plan} ({pct:.1f}%) | ETA {mm:02d}:{ss:02d}")

    total_dur = time.perf_counter() - t_total0
    if processed:
        print(f"\n{'='*60}")
        print(f"Total shots: {processed}/{plan}")
        print(f"Total time: {total_dur:.2f}s ({total_dur/60:.2f}m)")
        print(f"Avg/shot: {total_dur/processed:.2f}s")
        print(f"{'='*60}\n")
    return shotlist

def annotate_scene(scene_id: str, shots: List[Dict]) -> str:
    """Minimal scene summarizer placeholder."""
    return f"Scene {scene_id} summary."

def annotate_scenes(shotlist: List[Dict]) -> List[Dict]:
    """Apply a simple per-scene summary across shots."""
    scenes = get_unique_scenes(shotlist)
    for sid in scenes:
        scene_shots = [s for s in shotlist if (s.get('Scene') or '').strip() == sid]
        summary = annotate_scene(sid, scene_shots)
        for s in scene_shots:
            s['Scene_Caption'] = summary
    return shotlist