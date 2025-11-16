# PySceneDetect-based shot & scene detection.
# Requires: pip install scenedetect

from typing import List, Dict, Optional
import os

try:
    from scenedetect import VideoManager, SceneManager, StatsManager
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    from scenedetect.frame_timecode import FrameTimecode
except ImportError:
    VideoManager = None

def _ensure(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if VideoManager is None:
        raise RuntimeError("scenedetect not installed (pip install scenedetect)")

def _format_tc(ftc: FrameTimecode) -> str:
    return ftc.get_timecode()  # HH:MM:SS.mmm

def _parse_window_time(t: Optional[str], base: FrameTimecode) -> Optional[FrameTimecode]:
    if not t:
        return None
    if ":" in t:
        return FrameTimecode(timecode=t, fps=base.framerate)
    return FrameTimecode(timecode=float(t), fps=base.framerate)

def detect_shots(
    video_path: str,
    method: str = "adaptive",
    threshold: Optional[float] = 3.0,          # DEFAULT CHANGED: explicit 3.0
    min_scene_len_frames: int = 12,
    downscale_factor: int = 2,
    luma_only: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    AdaptiveDetector (PySceneDetect 0.6.x) does NOT support downscale_factor or luma_only.
    ContentDetector supports: threshold, min_scene_len, luma_only, downscale_factor, kernel_size.
    """
    _ensure(video_path)
    vm = VideoManager([video_path])
    vm.start()
    try:
        base_tc = vm.get_base_timecode()
        start_tc = _parse_window_time(start, base_tc) if start else None
        end_tc = _parse_window_time(end, base_tc) if end else None
        if start_tc or end_tc:
            vm.set_duration(start_time=start_tc, end_time=end_tc)

        stats = StatsManager()
        sm = SceneManager(stats)

        if method == "adaptive":
            det = AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=min_scene_len_frames
                # window_width / weights keep defaults
            )
        elif method == "content":
            det = ContentDetector(
                threshold=threshold,
                min_scene_len=min_scene_len_frames,
                luma_only=luma_only,
                downscale_factor=downscale_factor,
                kernel_size=3
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        sm.add_detector(det)
        sm.detect_scenes(vm)
        scenes = sm.get_scene_list()
    finally:
        vm.release()

    if not scenes:
        if verbose:
            print("No cuts detected; creating single shot.")
        # Fallback single shot covering requested window or full duration
        vm2 = VideoManager([video_path])
        vm2.start()
        full_end = vm2.get_base_timecode() + vm2.get_duration()
        vm2.release()
        s = start_tc or base_tc
        e = end_tc or full_end
        scenes = [(s, e)]

    shots: List[Dict] = []
    for i, (s_tc, e_tc) in enumerate(scenes):
        shots.append({
            "Ignore": "",
            "Scene": "",
            "Start": _format_tc(s_tc),
            "End": _format_tc(e_tc),
            "Shot_Caption": "",
            "Scene_Caption": ""
        })
        if verbose:
            dur = e_tc.get_seconds() - s_tc.get_seconds()
            print(f"  [SHOT {i}] {s_tc.get_seconds():.3f}s → {e_tc.get_seconds():.3f}s ({dur:.3f}s)")

    if verbose:
        print(f"Detected {len(shots)} shots.")
    return shots

def detect_scenes(
    shotlist: List[Dict],
    gap_sec: float = 5.0,
    min_scene_shots: int = 5,
    verbose: bool = False
) -> List[Dict]:
    """
    Simple gap-based scene grouping atop existing shotlist.
    """
    def _parse_tc(tc: str) -> float:
        if not tc:
            return 0.0
        hh, mm, rest = tc.split(":")
        ss = float(rest)
        return int(hh)*3600 + int(mm)*60 + ss

    if not shotlist:
        return shotlist

    current = 1
    count_in = 0
    last_end = _parse_tc(shotlist[0].get("End","0"))
    for shot in shotlist:
        start = _parse_tc(shot.get("Start","0"))
        end = _parse_tc(shot.get("End","0"))
        gap = start - last_end
        if gap >= gap_sec and count_in >= min_scene_shots:
            current += 1
            count_in = 0
            if verbose:
                print(f"  [SCENE CUT] gap={gap:.2f}s → Scene {current}")
        shot["Scene"] = str(current)
        count_in += 1
        last_end = max(last_end, end)

    if verbose:
        print(f"Assigned {current} scenes.")
    return shotlist
