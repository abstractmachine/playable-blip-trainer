import os
from datetime import datetime
from typing import List, Dict, Optional
from ollama import OllamaClient
import time  # fix: needed for timing
import json  # fix: used for load/dumps
import cv2   # fix: used for frame extraction

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
    Load system.txt from repo (fallback to project_root/prompts/system.txt), then minify.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "system.txt")
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(project_root, "prompts", "system.txt")

    if not os.path.exists(prompt_path):
        return ""

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    prompt = prompt.replace("{title}", film.get('title') or film.get('Title', 'Unknown'))
    prompt = prompt.replace("{year}", str(film.get('year', 'Unknown')))
    prompt = prompt.replace("{director}", film.get('director', 'Unknown'))
    prompt = prompt.replace("{image-count}", str(image_count))

    return _minify_system_text(prompt)

def has_scenes(shotlist: List[Dict]) -> bool:
    for shot in shotlist:
        if 'Scene' in shot and shot['Scene'].strip():
            return True
    return False

def get_unique_scenes(shotlist: List[Dict]) -> List[str]:
    scenes = set()
    for shot in shotlist:
        if 'Scene' in shot and shot['Scene'].strip():
            scenes.add(shot['Scene'].strip())
    return sorted(list(scenes), key=lambda x: int(x) if x.isdigit() else 0)

def parse_timecode(tc: str) -> float:
    parts = tc.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h)*3600 + int(m)*60 + float(s)
    return 0.0

def extract_frame_at_time(video_path: str, timestamp: float, output_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ret, frame = cap.read()
    if ret and frame is not None:
        cv2.imwrite(output_path, frame)
        cap.release()
        return True
    cap.release()
    return False

def extract_frames_for_shot(video_path: str, start_tc: str, end_tc: str, output_dir: str, movie_base: str, shot_index: int) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    start = parse_timecode(start_tc)
    end = parse_timecode(end_tc)
    dur = end - start
    if dur <= 0:
        return []
    segment = dur / 7.0
    timestamps = [start + segment * i for i in range(2, 7)]
    paths = []
    for i, ts in enumerate(timestamps):
        img_path = os.path.join(output_dir, f"{movie_base}_shot_{shot_index:04d}_frame_{i:02d}.png")
        if os.path.exists(img_path):
            # no noisy prints
            paths.append(img_path)
        else:
            if extract_frame_at_time(video_path, ts, img_path):
                # no noisy prints
                paths.append(img_path)
    return paths

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
    print_prompt: bool = False,
    ndjson_path: Optional[str] = None,
    verbose: bool = False
) -> tuple[str, float, Dict[str, float]]:
    """
    Annotate a single shot with Shot_Caption.
    
    Returns:
        (caption, total_duration_seconds, timings_dict)
    """
    t_shot0 = time.perf_counter()

    # Handle ignore/invalid TCs early
    if shot.get('Ignore', '').strip().lower() == 'yes':
        shot['Shot_Caption'] = ""
        return "", 0.0, {"images": 0, "frame_extract_s": 0.0, "ollama_s": 0.0, "parse_s": 0.0, "total_s": 0.0}
    start_tc = shot.get('Start', '')
    end_tc = shot.get('End', '')
    if not start_tc or not end_tc:
        shot['Shot_Caption'] = ""
        return "", 0.0, {"images": 0, "frame_extract_s": 0.0, "ollama_s": 0.0, "parse_s": 0.0, "total_s": 0.0}

    # Extract frames timing
    t0 = time.perf_counter()
    movie_filename = (film.get('filename') or film.get('Filename') or '')
    movie_base = os.path.splitext(movie_filename)[0]
    image_paths = extract_frames_for_shot(video_path, start_tc, end_tc, frames_dir, movie_base, index-1)
    frame_extract_s = time.perf_counter() - t0

    image_count = len(image_paths)
    if image_count == 0:
        shot['Shot_Caption'] = ""
        total_s = time.perf_counter() - t_shot0
        return "", total_s, {"images": 0, "frame_extract_s": frame_extract_s, "ollama_s": 0.0, "parse_s": 0.0, "total_s": total_s}

    # System prompt
    system_text = load_system_prompt(project_root, image_count, film)

    # Model call timing
    user_prompt = (
        "Respond ONLY with a JSON object that matches the provided schema. "
        "Do not include prose, markdown, code fences, keys outside the schema, or comments."
    )
    schema = load_annotation_schema(project_root)

    t1 = time.perf_counter()
    response = ollama.generate_with_images(
        prompt=user_prompt,
        image_paths=image_paths,
        stream=False,
        system=system_text,
        schema=schema
    )
    ollama_s = time.perf_counter() - t1

    # Parse timing
    parse_s = 0.0
    if response is None:
        shot['Shot_Caption'] = ""
        total_s = time.perf_counter() - t_shot0
        return "", total_s, {"images": image_count, "frame_extract_s": frame_extract_s, "ollama_s": ollama_s, "parse_s": 0.0, "total_s": total_s}

    try:
        t2 = time.perf_counter()
        data = json.loads(response)
        data = _ensure_list_fields(data)
        shot['Shot_Caption'] = _minify(data)
        parse_s = time.perf_counter() - t2

        if ndjson_path:
            audit = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "film": {
                    "title": film.get("title") or film.get("Title"),
                    "year": film.get("year"),
                    "filename": film.get("filename") or film.get("Filename"),
                },
                "shot_index": index,
                "start": start_tc,
                "end": end_tc,
                "frames": [os.path.basename(p) for p in image_paths],
                "timings": {
                    "images": image_count,
                    "frame_extract_s": round(frame_extract_s, 4),
                    "ollama_s": round(ollama_s, 4),
                    "parse_s": round(parse_s, 4),
                },
                "output": data
            }
            with open(ndjson_path, "a", encoding="utf-8") as f:
                f.write(_minify(audit) + "\n")
    except Exception as e:
        print(f"[warn] JSON parse/validate failed for shot {index}: {e}")
        shot['Shot_Caption'] = ""

    total_s = time.perf_counter() - t_shot0
    timings = {
        "images": image_count,
        "frame_extract_s": frame_extract_s,
        "ollama_s": ollama_s,
        "parse_s": parse_s,
        "total_s": total_s
    }
    if verbose:
        print(f"    details: frames={image_count} | extract {frame_extract_s:.2f}s | model {ollama_s:.2f}s | parse {parse_s:.2f}s")
    return shot['Shot_Caption'], total_s, timings

def annotate_shots(
    shotlist: List[Dict],
    video_path: str,
    film: Dict,
    ollama: OllamaClient,
    frames_dir: str,
    project_root: str,
    limit: Optional[int] = None,
    start_index: int = 1,
    verbose: bool = False
) -> List[Dict]:
    """
    Annotate each shot with Shot_Caption.

    Args:
        limit: number of shots to process (None = until end)
        start_index: 1-based index of first shot to process
        verbose: print detailed timing and ETA
    """
    os.makedirs(frames_dir, exist_ok=True)
    ndjson_path = os.path.join(
        frames_dir,
        f"{os.path.splitext((film.get('filename') or film.get('Filename') or 'unknown'))[0]}.annotations.ndjson"
    )

    total = len(shotlist)
    planned_total = (min(limit, max(0, total - (start_index - 1))) if limit is not None else max(0, total - (start_index - 1)))
    print(f"Starting annotation: {planned_total} planned shots (from {start_index} to {total})")

    processed = 0
    sum_duration = 0.0
    total_start_time = time.perf_counter()
    
    for i, shot in enumerate(shotlist, start=1):
        # Skip shots before the requested starting index
        if i < max(1, start_index):
            continue
        if limit is not None and processed >= limit:
            break

        # Skip ignored shots but do not count toward limit
        if shot.get('Ignore', '').strip().lower() == 'yes':
            shot['Shot_Caption'] = ""
            continue

        print(f"Processing shot {i}/{total}...")
        caption, duration, timings = annotate_shot(
            shot, i, video_path, film, ollama, frames_dir, project_root,
            print_prompt=False, ndjson_path=ndjson_path, verbose=verbose
        )
        if caption is not None:
            shot['Shot_Caption'] = caption
        
        processed += 1
        sum_duration += duration
        avg_duration = (sum_duration / processed) if processed else 0.0
        remaining = (planned_total - processed) if planned_total else 0
        eta_seconds = max(0.0, remaining * avg_duration)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        pct = (processed / planned_total * 100.0) if planned_total else 100.0

        print(f"  ✓ Completed in {duration:.2f}s | progress {processed}/{planned_total} ({pct:.1f}%) | ETA {eta_min:02d}:{eta_sec:02d}")
        if verbose:
            print(f"    breakdown: extract {timings['frame_extract_s']:.2f}s | model {timings['ollama_s']:.2f}s | parse {timings['parse_s']:.2f}s")

    total_duration = time.perf_counter() - total_start_time
    
    if processed > 0:
        avg_duration = total_duration / processed
        print(f"\n{'='*60}")
        print(f"Annotation Summary:")
        print(f"  Total shots annotated: {processed}/{planned_total}")
        print(f"  Total time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        print(f"  Average time per shot: {avg_duration:.2f}s")
        print(f"{'='*60}\n")

    return shotlist

def annotate_scene(scene_id: str, shots: List[Dict]) -> str:
    return ""

def annotate_scenes(shotlist: List[Dict]) -> List[Dict]:
    if not has_scenes(shotlist):
        raise ValueError("Cannot annotate scenes: No scene information found in shotlist")
    scenes = {}
    for shot in shotlist:
        scene_id = shot.get('Scene', '').strip()
        if scene_id:
            scenes.setdefault(scene_id, []).append(shot)
    for scene_id, shots in scenes.items():
        caption = annotate_scene(scene_id, shots)
        for shot in shots:
            shot['Scene_Caption'] = caption
    return shotlist