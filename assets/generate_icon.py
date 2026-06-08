"""Generate InvestBot.ico for Windows exe / tkinter window."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "invest_bot.ico"


def main():
    size = 256
    img = Image.new("RGBA", (size, size), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)

    # Rounded card background
    draw.rounded_rectangle((24, 24, 232, 232), radius=36, fill=(30, 41, 59, 255))

    # Upward chart line
    pts = [(52, 170), (96, 130), (128, 145), (168, 88), (204, 108)]
    draw.line(pts, fill=(52, 211, 153, 255), width=10, joint="curve")
    for x, y in pts:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(110, 231, 183, 255))

    # Accent bar
    draw.rounded_rectangle((52, 188, 204, 206), radius=8, fill=(59, 130, 246, 255))

    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    img.save(OUT, format="ICO", sizes=[(s, s) for s, _ in sizes])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
