"""Build labelled contact sheets from already rendered thesis and figure PNGs.

This is a layout-review helper; it does not alter source figures or numerical results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_sheet(
    paths: list[Path],
    destination: Path,
    *,
    columns: int,
    rows: int,
    tile_width: int,
    tile_height: int,
    title: str,
) -> None:
    margin = 18
    heading_height = 52
    cell_width = tile_width + 2 * margin
    cell_height = tile_height + 2 * margin + 28
    canvas = Image.new(
        "RGB", (columns * cell_width, heading_height + rows * cell_height), "#eceff2"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), title, fill="#16202b", font=font(24))

    for index, path in enumerate(paths):
        column = index % columns
        row = index // columns
        x = column * cell_width + margin
        y = heading_height + row * cell_height + margin
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (tile_width, tile_height))
        px = x + (tile_width - image.width) // 2
        py = y + (tile_height - image.height) // 2
        draw.rectangle((x - 1, y - 1, x + tile_width, y + tile_height), fill="white")
        canvas.paste(image, (px, py))
        draw.rectangle((x - 1, y - 1, x + tile_width, y + tile_height), outline="#abb5bf")
        draw.text((x, y + tile_height + 5), path.stem, fill="#16202b", font=font(17))

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def chunks(paths: list[Path], size: int) -> list[list[Path]]:
    return [paths[start : start + size] for start in range(0, len(paths), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pages = sorted(
        args.pages.glob("page-*.png"), key=lambda item: int(item.stem.rsplit("-", 1)[1])
    )
    figures = sorted(args.figures.glob("*.png"))
    if not pages or not figures:
        raise SystemExit("Both rendered pages and rendered figures are required")

    for number, group in enumerate(chunks(pages, 4), start=1):
        draw_sheet(
            group,
            args.output / f"pages_{number:02d}.png",
            columns=2,
            rows=2,
            tile_width=560,
            tile_height=790,
            title=f"Thesis pages {group[0].stem} through {group[-1].stem}",
        )

    for number, group in enumerate(chunks(figures, 2), start=1):
        draw_sheet(
            group,
            args.output / f"figures_{number:02d}.png",
            columns=2,
            rows=1,
            tile_width=780,
            tile_height=590,
            title=f"Source figures {2 * number - 1} and {min(2 * number, len(figures))}",
        )

    print(f"{len(pages)} pages in {(len(pages) + 3) // 4} contact sheets")
    print(f"{len(figures)} figures in {(len(figures) + 1) // 2} contact sheets")


if __name__ == "__main__":
    main()
