"""Build offline replay plus GIF and optional MP4 files from reviewed evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1120, 580


def font(size, bold=False, mono=False):
    for name in (
        ("C:/Windows/Fonts/consola.ttf" if mono else
         "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        "DejaVuSansMono.ttf" if mono else "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise SystemExit("Install Arial or DejaVu Sans to render the replay.")


def wrapped(draw, value, x, y, limit, face, color, line_height):
    line = ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=face) > limit:
            draw.text((x, y), line, font=face, fill=color)
            line, y = word, y + line_height
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=face, fill=color)


def render(case, index):
    name, _, *details = case["beats"][index]
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0a1220")
    canvas = ImageDraw.Draw(image)
    canvas.text((36, 30), f"τ²-BENCH   /   {case['domain'].upper()}", font=font(20, bold=True), fill="#a7dcff")
    canvas.text((36, 66), case["en"], font=font(41, bold=True), fill="#eef4ff")
    canvas.text((36, 131), name, font=font(25, bold=True), fill="#b9cbe3")
    for column, label in enumerate(("Original Qwen3.5-2B", "SFT + GRPO + static playbooks")):
        left = 36 + 532 * column
        accent = ("#ffb3a7", "#9cebd3")[column]
        tool, note = details[2 * column:2 * column + 2]
        canvas.rounded_rectangle((left, 190, left + 516, 473), radius=15, fill="#142238", outline="#465a73", width=2)
        canvas.text((left + 20, 213), label, font=font(27, bold=True), fill=accent)
        result = ("FAIL · Reward 0", "PASS · Reward 1")[column]
        canvas.text((left + 20, 254), f"{result} · {case['steps'][column]} steps", font=font(19), fill="#b9cbe3")
        canvas.text((left + 20, 306), tool, font=font(19, mono=True), fill="#a7dcff")
        wrapped(canvas, note, left + 20, 349, 472, font(23), "#eef4ff", 30)
    for i in range(len(case["beats"])):
        x = 36 + i * 74
        canvas.rounded_rectangle((x, 510, x + 62, 517), radius=3, fill="#a7dcff" if i == index else "#3a4d65")
    canvas.text((36, 532), "Abridged trace replay · Matched task, simulator and scoring", font=font(17), fill="#9cafc4")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    cases = json.loads((ROOT / "data/cases.json").read_text(encoding="utf-8"))
    dest = ROOT / "demos"
    dest.mkdir(exist_ok=True)
    template = (ROOT / "tools/replay.template.html").read_text(encoding="utf-8")
    embedded = json.dumps(cases, ensure_ascii=False).replace("<", "\\u003c")
    (dest / "index.html").write_text(template.replace("__CASES__", embedded), encoding="utf-8")
    for case in cases:
        frames = [render(case, index) for index in range(len(case["beats"]))]
        frames[-1].save(dest / f"{case['id']}.png", optimize=True)
        palette = [frame.quantize(colors=128) for frame in frames]
        palette[0].save(dest / f"{case['id']}.gif", save_all=True, append_images=palette[1:],
                        duration=[2100, 2800, 3400, 3700], loop=0, disposal=2)
        if args.ffmpeg:
            with TemporaryDirectory() as tmp:
                folder = Path(tmp)
                index = 0
                for frame, seconds in zip(frames, (2, 3, 4, 4)):
                    for _ in range(seconds):
                        frame.save(folder / f"{index:03}.png")
                        index += 1
                subprocess.run([args.ffmpeg, "-v", "error", "-y", "-framerate", "1",
                    "-i", str(folder / "%03d.png"), "-c:v", "libx264", "-r", "24",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest / f"{case['id']}.mp4")],
                    check=True)


if __name__ == "__main__":
    main()
