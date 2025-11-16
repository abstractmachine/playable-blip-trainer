import cv2
import os
from typing import List, Dict, Optional

def _tc(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def _parse_tc(tc: str) -> float:
    if not tc:
        return 0.0
    parts = tc.split(":")
    try:
        if len(parts) == 3:
            hh = int(parts[0]); mm = int(parts[1]); ss = float(parts[2])
        elif len(parts) == 2:
            hh = 0; mm = int(parts[0]); ss = float(parts[1])
        else:
            return float(tc)
        return hh*3600 + mm*60 + ss
    except Exception:
        return 0.0

def detect_shots(
    video_path: str,
    threshold: float = 0.6,
    stride: int = 12,
    min_shot_sec: float = 0.5,
    verbose: bool = False
) -> List[Dict]:
    """
    Naive shot boundary detection using HSV histogram distance.
    Writes a shot list with Start/End timecodes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0

    def hist(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0,1], None, [50,60], [0,180, 0,256])
        return cv2.normalize(h, h).flatten()

    shots: List[Dict] = []
    prev_hist = None
    cut_times: List[float] = [0.0]

    frame_idx = 0
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return []
    prev_hist = hist(frame)
    frame_idx += 1

    while True:
        # Skip stride-1 frames
        for _ in range(stride-1):
            ok = cap.grab()
            if not ok:
                break
            frame_idx += 1
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break
        frame_idx += 1

        h = hist(frame)
        dist = cv2.compareHist(prev_hist, h, cv2.HISTCMP_BHATTACHARYYA)
        prev_hist = h

        if dist >= threshold:
            t = min((frame_idx-1) / fps, max(0.0, duration - 1.0/fps))
            # Respect min shot length
            if t - cut_times[-1] >= min_shot_sec:
                cut_times.append(t)
                if verbose:
                    print(f"    [CUT] {t:.3f}s (dist={dist:.3f})")

        if frame_count and frame_idx >= frame_count:
            break

    cap.release()
    # Close last shot
    if not cut_times or cut_times[-1] < duration:
        cut_times.append(duration)

    # Assemble shot rows
    for i in range(len(cut_times)-1):
        start_s = cut_times[i]
        end_s = cut_times[i+1]
        if end_s - start_s < 1e-3:
            continue
        shots.append({
            "Start": _tc(start_s),
            "End": _tc(end_s),
            "Ignore": "",
            "Shot_Caption": "",
            "Scene": "",
            "Scene_Caption": ""
        })

    if verbose:
        print(f"Detected {len(shots)} shots (duration {duration:.2f}s, fps {fps:.2f})")
    return shots

def detect_scenes(
    shotlist: List[Dict],
    gap_sec: float = 5.0,
    min_scene_shots: int = 5,
    verbose: bool = False
) -> List[Dict]:
    """
    Naive scene grouping:
    - Start Scene 1 at first shot
    - Increment scene when time gap between previous End and current Start >= gap_sec
    - Ensure minimum scene size by delaying increment if needed
    """
    if not shotlist:
        return shotlist

    current_scene = 1
    scene_count = 1
    shots_in_scene = 0
    last_end = _parse_tc(shotlist[0].get("End","0"))

    for i, shot in enumerate(shotlist):
        start = _parse_tc(shot.get("Start","0"))
        end = _parse_tc(shot.get("End","0"))

        gap = max(0.0, start - last_end)
        should_cut = gap >= gap_sec and shots_in_scene >= min_scene_shots

        if should_cut:
            current_scene += 1
            shots_in_scene = 0
            if verbose:
                print(f"    [SCENE CUT] gap={gap:.2f}s -> Scene {current_scene}")

        shot["Scene"] = str(current_scene)
        shots_in_scene += 1
        last_end = max(last_end, end)

    if verbose:
        print(f"Assigned {current_scene} scenes.")
    return shotlist
