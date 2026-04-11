"""Generate a simple drag-to-Applications DMG background image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 640
HEIGHT = 360


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_background(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGBA", (WIDTH, HEIGHT), "#f6efe4")
    draw = ImageDraw.Draw(image)

    for y in range(HEIGHT):
        ratio = y / max(1, HEIGHT - 1)
        shade = int(246 - (18 * ratio))
        draw.line([(0, y), (WIDTH, y)], fill=(shade, shade - 6, shade - 16, 255))

    title_font = _load_font(28)
    subtitle_font = _load_font(18)
    caption_font = _load_font(16)

    draw.rounded_rectangle((48, 96, 208, 256), radius=28, fill="#fff9f2", outline="#d7c6af", width=3)
    draw.rounded_rectangle((432, 96, 592, 256), radius=28, fill="#fff9f2", outline="#d7c6af", width=3)
    draw.text((60, 36), "Install MuFinder", fill="#2c2418", font=title_font)
    draw.text((60, 72), "Drag the app into Applications", fill="#7a6b59", font=subtitle_font)

    arrow_points = [
        (248, 176),
        (366, 176),
        (366, 150),
        (422, 190),
        (366, 230),
        (366, 204),
        (248, 204),
    ]
    draw.polygon(arrow_points, fill="#e02020")

    draw.text((78, 272), "MuFinder.app", fill="#2c2418", font=caption_font)
    draw.text((452, 272), "Applications", fill="#2c2418", font=caption_font)

    image.save(output_path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Destination PNG path")
    args = parser.parse_args()
    build_background(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
