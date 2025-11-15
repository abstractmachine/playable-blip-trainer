import os
from data import Cinematheque, Gameplay
from annotate import annotate_shots, annotate_scenes, has_scenes
from ollama import OllamaClient
from parse import parse_arguments

def main():
    args = parse_arguments()

    # Library by media
    if args.media == 'gameplay':
        metadata_path = os.path.join(args.project_root, "metadata", "gameplay.csv")
        library = Gameplay(metadata_path, args.project_root)
        media_type = "gameplay"
    else:
        metadata_path = os.path.join(args.project_root, "metadata", "cinematheque.csv")
        library = Cinematheque(metadata_path, args.project_root)
        media_type = "film"

    print(f"Loaded {library}")

    ollama = OllamaClient(model=args.model, num_ctx=args.num_ctx)

    def process_index(idx: int):
        item = library.get(idx)
        if not item:
            print(f"Index {idx} out of range (0-{len(library)-1})")
            return
        print(f"\n{media_type.capitalize()} [{idx}]: {library.get_title(item)}")

        if args.action == 'erase':
            if args.type == 'shot':
                print("Erasing Shot_Caption entries...")
                ok = library.erase_shot_captions(item)
                print("✓ Done" if ok else "✗ Failed")
            elif args.type == 'scene':
                print("Erasing Scene_Caption entries...")
                ok = library.erase_scene_captions(item)
                print("✓ Done" if ok else "✗ Failed")
            return

        if args.action == 'annotate':
            shotlist = library.load_shotlist(item)
            if not shotlist:
                print("✗ No shotlist found")
                return
            if args.type == 'shot':
                video_path = library.get_video_path(item)
                if not os.path.exists(video_path):
                    print(f"✗ Video not found: {video_path}")
                    return
                frames_dir = os.path.join(args.project_root, "frames")
                out = annotate_shots(
                    shotlist=shotlist,
                    video_path=video_path,
                    film=item,
                    ollama=ollama,
                    frames_dir=frames_dir,
                    project_root=args.project_root,
                    limit=args.annotation_count,
                    start_index=args.shot_index,
                    verbose=args.verbose,
                    save_callback=lambda sl: library.save_shotlist(item, sl)  # Save after each shot
                )
                if library.save_shotlist(item, out):
                    print("✓ Saved shot annotations")
                else:
                    print("✗ Failed to save")
            elif args.type == 'scene':
                if not has_scenes(shotlist):
                    print("✗ No scene info available")
                    return
                out = annotate_scenes(shotlist)
                if library.save_shotlist(item, out):
                    print("✓ Saved scene annotations")
                else:
                    print("✗ Failed to save")
            return

        # info-only
        shotlist = library.load_shotlist(item)
        if shotlist:
            print(f"Shotlist: {len(shotlist)} shots | Scenes: {'yes' if has_scenes(shotlist) else 'no'}")
        else:
            print("No shotlist found")

    # Batch mode via --filelist
    if args.filelist:
        path = args.filelist if os.path.isabs(args.filelist) else os.path.join(os.getcwd(), args.filelist)
        if not os.path.exists(path):
            print(f"✗ Filelist not found: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
        print(f"\nProcessing {len(names)} items from filelist ({media_type})...")
        for name in names:
            idx = library.find_by_filename(name)
            if idx is None:
                print(f"— Skipping (not in metadata): {name}")
                continue
            process_index(idx)
        return

    # Single selection
    if args.index >= 0:
        process_index(args.index)
    else:
        print(f"\nAvailable {media_type} items:")
        print("=" * 60)
        for i, item in enumerate(library.items):
            title = library.get_title(item)
            print(f"{i}\t{title}")
        print("=" * 60)
        print(f"Total: {len(library.items)} items")

if __name__ == "__main__":
    main()