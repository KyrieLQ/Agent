"""Render the same product replay UI to offline HTML, GIF, PNG, and MP4."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1280, 700
HOLDS = (2, 3, 4, 4)


def screenshot(browser, url, output, profile):
    command = [
        browser, "--headless=new", "--disable-background-networking",
        "--no-first-run", "--disable-gpu", "--hide-scrollbars",
        f"--user-data-dir={profile}", f"--window-size={WIDTH},{HEIGHT}",
        "--force-device-scale-factor=1", f"--screenshot={output}", url,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=45)
    deadline = time.monotonic() + 12
    while not output.is_file() and time.monotonic() < deadline:
        time.sleep(0.15)
    with Image.open(output) as frame:
        if frame.size != (WIDTH, HEIGHT):
            raise ValueError(f"Unexpected screenshot dimensions: {frame.size}")
        return frame.convert("RGB")


def render_media(case_id, dest, browser, ffmpeg, readme_frames=False):
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        frames = [
            screenshot(browser, f"{(dest / 'index.html').as_uri()}?capture=1&case={case_id}&step={index}",
                       folder / f"shot-{index}.png", folder / "profile")
            for index in range(4)
        ]
        frames[-1].save(dest / f"{case_id}.png", optimize=True)
        if readme_frames:
            frame_dir = dest / "frames"
            frame_dir.mkdir(exist_ok=True)
            for number, frame in enumerate(frames[:-1], start=1):
                frame.save(frame_dir / f"{case_id}-{number}.png", optimize=True)
            return
        palette = [frame.quantize(colors=150) for frame in frames]
        palette[0].save(dest / f"{case_id}.gif", save_all=True, append_images=palette[1:],
                        duration=[s * 1000 for s in HOLDS], loop=0, disposal=2)
        if ffmpeg:
            number = 0
            for frame, seconds in zip(frames, HOLDS):
                for _ in range(seconds):
                    frame.save(folder / f"{number:03}.png")
                    number += 1
            subprocess.run([ffmpeg, "-v", "error", "-y", "-framerate", "1",
                "-i", str(folder / "%03d.png"), "-c:v", "libx264", "-r", "24",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest / f"{case_id}.mp4")],
                check=True, timeout=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", help="Path to Chrome, Edge, or Chromium for image/video export")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--readme-frames", action="store_true", help="Export only the README's static frames")
    args = parser.parse_args()
    if args.readme_frames and not args.browser:
        parser.error("--readme-frames requires --browser")
    cases = json.loads((ROOT / "data/cases.json").read_text(encoding="utf-8"))
    scenes = json.loads((ROOT / "data/product-replay.json").read_text(encoding="utf-8"))
    if {case["id"] for case in cases} != scenes.keys():
        raise ValueError("Every case needs a matching product view")
    dest = ROOT / "demos"
    dest.mkdir(exist_ok=True)
    template = (ROOT / "tools/product.template.html").read_text(encoding="utf-8")
    data = json.dumps(dict(cases=cases, scenes=scenes), ensure_ascii=False).replace("<", "\\u003c")
    (dest / "index.html").write_text(template.replace("__REPLAY_DATA__", data), encoding="utf-8")
    if args.browser:
        for case in cases:
            render_media(case["id"], dest, args.browser, args.ffmpeg, args.readme_frames)
            print(f"Rendered {case['domain']} comparison replay.")


if __name__ == "__main__":
    main()
