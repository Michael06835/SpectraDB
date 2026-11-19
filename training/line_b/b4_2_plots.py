"""Pillow-based figure rendering for the B4.2 analysis.

Matplotlib's native renderer crashes in this project environment (delay-load
DLL fault during ``savefig``), so the B4.2 figures are drawn with Pillow.
Only PNG output, fixed fonts, deterministic layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("C:/Windows/Fonts")
_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _FONT_CACHE:
        candidates = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
        for name in candidates:
            path = FONT_DIR / name
            if path.is_file():
                _FONT_CACHE[key] = ImageFont.truetype(str(path), size)
                break
        else:  # pragma: no cover - defensive
            raise RuntimeError("No usable TrueType font found under C:/Windows/Fonts")
    return _FONT_CACHE[key]


_PALETTES = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "magma": [(0, 0, 4), (80, 18, 123), (182, 54, 121), (252, 137, 97), (252, 253, 191)],
    "cividis": [(0, 32, 76), (55, 76, 110), (126, 124, 109), (188, 178, 96), (255, 233, 69)],
}
COLORS = {
    "ir": (31, 119, 180),
    "raman": (214, 39, 40),
    "uvvis": (44, 160, 44),
    "Shared14": (31, 119, 180),
    "UVSPEC15": (255, 127, 14),
    "seeds": [(31, 119, 180), (214, 39, 40), (44, 160, 44)],
}
TEXT = (35, 35, 35)
GRID = (210, 210, 210)


def palette_at(name: str, t: float) -> tuple[int, int, int]:
    stops = _PALETTES.get(name, _PALETTES["viridis"])
    t = min(1.0, max(0.0, float(t)))
    position = t * (len(stops) - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(stops) - 1)
    fraction = position - lower
    return tuple(
        int(round(stops[lower][channel] * (1 - fraction) + stops[upper][channel] * fraction))
        for channel in range(3)
    )


def _luminance(color: tuple[int, int, int]) -> float:
    red, green, blue = color
    return (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0


class Canvas:
    def __init__(self, width: int, height: int, background: tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), background)
        self.draw = ImageDraw.Draw(self.image)

    def text(self, xy, text, size=14, bold=False, fill=TEXT, anchor="la"):
        self.draw.text((int(xy[0]), int(xy[1])), str(text), font=_font(size, bold), fill=fill, anchor=anchor)

    def rect(self, box, fill=None, outline=None, width=1):
        self.draw.rectangle([int(value) for value in box], fill=fill, outline=outline, width=width)

    def line(self, points, fill=TEXT, width=1):
        self.draw.line([(int(x), int(y)) for x, y in points], fill=fill, width=width)

    def rotated_label(self, center_x: float, top_y: float, text: str, size=10, angle=55, fill=TEXT) -> int:
        font = _font(size)
        bbox = font.getbbox(str(text))
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        layer = Image.new("RGBA", (text_width + 10, text_height + 10), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((5, 3), str(text), font=font, fill=fill + (255,))
        rotated = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        left = int(center_x - rotated.width / 2)
        top = int(top_y)
        self.image.paste(rotated, (left, top), rotated)
        return rotated.height

    def vertical_label(self, x: float, center_y: float, text: str, size=13, fill=TEXT):
        font = _font(size)
        bbox = font.getbbox(str(text))
        layer = Image.new("RGBA", (bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 10), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((5, 3), str(text), font=font, fill=fill + (255,))
        rotated = layer.rotate(90, expand=True, resample=Image.BICUBIC)
        self.image.paste(rotated, (int(x - rotated.width / 2), int(center_y - rotated.height / 2)), rotated)

    def save(self, path: Path) -> None:
        self.image.save(path, "PNG")


def draw_colorbar(canvas: Canvas, box, vmin: float, vmax: float, palette: str, ticks: int = 5) -> None:
    x0, y0, x1, y1 = box
    steps = max(2, int(y1 - y0))
    for step in range(steps):
        fraction = 1.0 - step / (steps - 1)
        color = palette_at(palette, fraction)
        canvas.line([(x0, y0 + step), (x1, y0 + step)], fill=color)
    canvas.rect((x0, y0, x1, y1), outline=(90, 90, 90), width=1)
    for index in range(ticks + 1):
        value = vmin + (vmax - vmin) * index / ticks
        y = y1 - (y1 - y0) * index / ticks
        canvas.line([(x1, y), (x1 + 6, y)], fill=(90, 90, 90), width=1)
        canvas.text((x1 + 10, y), f"{value:.1f}", size=11, anchor="lm")


def draw_heatmap(
    canvas: Canvas,
    box,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    palette: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate: bool = True,
    annotate_size: int = 10,
    row_label_size: int = 13,
    col_label_size: int = 10,
    row_label_gap: int = 10,
    rotate_cols: bool = True,
) -> None:
    x0, y0, x1, y1 = box
    rows, cols = matrix.shape
    cell_width = (x1 - x0) / cols
    cell_height = (y1 - y0) / rows
    for row in range(rows):
        for col in range(cols):
            value = float(matrix[row, col])
            fraction = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.0
            color = palette_at(palette, fraction)
            left = x0 + col * cell_width
            top = y0 + row * cell_height
            canvas.rect((left, top, left + cell_width, top + cell_height), fill=color)
            if annotate:
                text_color = (255, 255, 255) if _luminance(color) < 0.55 else (15, 15, 15)
                canvas.text(
                    (left + cell_width / 2, top + cell_height / 2),
                    f"{value:.3f}",
                    size=annotate_size,
                    fill=text_color,
                    anchor="mm",
                )
    canvas.rect(box, outline=(80, 80, 80), width=1)
    for row, label in enumerate(row_labels):
        canvas.text((x0 - row_label_gap, y0 + (row + 0.5) * cell_height), label, size=row_label_size, anchor="rm")
    for col, label in enumerate(col_labels):
        center = x0 + (col + 0.5) * cell_width
        if rotate_cols:
            canvas.rotated_label(center, y1 + 6, label, size=col_label_size)
        else:
            canvas.text((center, y1 + 8), label, size=col_label_size, anchor="ma")


def draw_grouped_bars(
    canvas: Canvas,
    box,
    group_labels: list[str],
    series: list[dict],
    title: str,
    ylabel: str = "Test score",
    ymin: float = 0.0,
    ymax: float = 1.0,
    ticks: int = 5,
    legend: bool = True,
) -> None:
    x0, y0, x1, y1 = box
    canvas.rect((x0, y0, x1, y1), outline=(120, 120, 120), width=1)
    for index in range(ticks + 1):
        value = ymin + (ymax - ymin) * index / ticks
        y = y1 - (y1 - y0) * index / ticks
        canvas.line([(x0, y), (x1, y)], fill=GRID, width=1)
        canvas.text((x0 - 8, y), f"{value:.1f}", size=11, anchor="rm")
    canvas.vertical_label(x0 - 58, (y0 + y1) / 2, ylabel, size=13)
    group_width = (x1 - x0) / len(group_labels)
    bar_width = group_width * 0.72 / len(series)
    for group_index, group_label in enumerate(group_labels):
        group_left = x0 + group_index * group_width + group_width * 0.14
        for series_index, item in enumerate(series):
            value = float(item["values"][group_index])
            error = float(item.get("errors", [0.0] * len(group_labels))[group_index])
            left = group_left + series_index * bar_width
            top = y1 - (y1 - y0) * (value - ymin) / (ymax - ymin)
            color = item["color"]
            canvas.rect((left, top, left + bar_width * 0.9, y1), fill=color, outline=(60, 60, 60), width=1)
            if error > 0:
                center = left + bar_width * 0.45
                error_top = y1 - (y1 - y0) * (value + error - ymin) / (ymax - ymin)
                error_bottom = y1 - (y1 - y0) * (max(ymin, value - error) - ymin) / (ymax - ymin)
                canvas.line([(center, error_top), (center, error_bottom)], fill=(30, 30, 30), width=2)
                canvas.line([(center - 5, error_top), (center + 5, error_top)], fill=(30, 30, 30), width=2)
                canvas.line([(center - 5, error_bottom), (center + 5, error_bottom)], fill=(30, 30, 30), width=2)
        canvas.text((x0 + (group_index + 0.5) * group_width, y1 + 10), group_label, size=14, anchor="ma")
    canvas.text(((x0 + x1) / 2, y0 - 42), title, size=18, bold=True, anchor="ma")
    if legend:
        total_items = len(series)
        item_width = (x1 - x0) / total_items
        for item_index, item in enumerate(series):
            legend_x = x0 + item_index * item_width + 10
            legend_y = y1 + 46
            canvas.rect((legend_x, legend_y, legend_x + 14, legend_y + 14), fill=item["color"], outline=(60, 60, 60))
            canvas.text((legend_x + 20, legend_y + 7), item["label"], size=12, anchor="lm")


def draw_seed_panel(
    canvas: Canvas,
    box,
    modality_labels: list[str],
    seed_values: list[list[float]],
    means: list[float],
    sds: list[float],
    title: str,
    ymin: float,
    ymax: float,
) -> None:
    x0, y0, x1, y1 = box
    canvas.rect((x0, y0, x1, y1), outline=(120, 120, 120), width=1)
    for index in range(5):
        value = ymin + (ymax - ymin) * index / 4
        y = y1 - (y1 - y0) * index / 4
        canvas.line([(x0, y), (x1, y)], fill=GRID, width=1)
        canvas.text((x0 - 8, y), f"{value:.2f}", size=11, anchor="rm")
    group_width = (x1 - x0) / len(modality_labels)
    offsets = (-7, 0, 7)
    for group_index, label in enumerate(modality_labels):
        center = x0 + (group_index + 0.5) * group_width
        values = seed_values[group_index]
        for seed_index, value in enumerate(values):
            y = y1 - (y1 - y0) * (value - ymin) / (ymax - ymin)
            radius = 5
            canvas.draw.ellipse(
                (center + offsets[seed_index] - radius, y - radius, center + offsets[seed_index] + radius, y + radius),
                fill=COLORS["seeds"][seed_index],
                outline=(40, 40, 40),
            )
        mean = means[group_index]
        sd = sds[group_index]
        mean_y = y1 - (y1 - y0) * (mean - ymin) / (ymax - ymin)
        top_y = y1 - (y1 - y0) * (mean + sd - ymin) / (ymax - ymin)
        bottom_y = y1 - (y1 - y0) * (mean - sd - ymin) / (ymax - ymin)
        canvas.line([(center, top_y), (center, bottom_y)], fill=(20, 20, 20), width=2)
        canvas.line([(center - 8, top_y), (center + 8, top_y)], fill=(20, 20, 20), width=2)
        canvas.line([(center - 8, bottom_y), (center + 8, bottom_y)], fill=(20, 20, 20), width=2)
        canvas.line([(center - 10, mean_y), (center + 10, mean_y)], fill=(20, 20, 20), width=3)
        canvas.text((center, y1 + 10), label, size=14, anchor="ma")
    canvas.text(((x0 + x1) / 2, y0 - 40), title, size=16, bold=True, anchor="ma")


def _figure_legend(canvas: Canvas, x: int, y: int, items: list[tuple[str, tuple[int, int, int]]], title: str | None = None) -> None:
    if title:
        canvas.text((x, y), title, size=13, bold=True, anchor="la")
        y += 22
    for label, color in items:
        canvas.rect((x, y, x + 14, y + 14), fill=color, outline=(60, 60, 60))
        canvas.text((x + 20, y + 7), label, size=12, anchor="lm")
        y += 22


def build_overall_heatmap(path: Path, models: list[dict], metrics: list[str]) -> None:
    canvas = Canvas(1220, 1080)
    canvas.text((610, 24), "B4.2 Test performance by model (threshold = 0.5)", size=22, bold=True, anchor="ma")
    matrix = np.asarray([[model["overall"]["fixed_0.5"][metric] for metric in metrics] for model in models])
    draw_heatmap(
        canvas,
        (300, 130, 1010, 1000),
        matrix,
        [f'{model["task"]} {model["modality"].upper()} {model["seed"]}' for model in models],
        ["Micro-F1", "Macro-F1", "mAP", "Macro-AUROC"],
        vmin=0.0,
        vmax=1.0,
        annotate=True,
        annotate_size=10,
        row_label_size=12,
        col_label_size=12,
        rotate_cols=False,
    )
    draw_colorbar(canvas, (1040, 130, 1064, 1000), 0.0, 1.0, "viridis")
    canvas.save(path)


def build_f1_bars(path: Path, task_rows: dict) -> None:
    canvas = Canvas(1420, 720)
    canvas.text((710, 24), "B4.2 Test F1 by modality (threshold = 0.5; mean ± SD, n=3 seeds)", size=21, bold=True, anchor="ma")
    specs = [
        ("macro_f1", "Macro-F1", (110, 150, 670, 620)),
        ("micro_f1", "Micro-F1", (800, 150, 1360, 620)),
    ]
    for metric, title, box in specs:
        series = []
        for modality in ("ir", "raman", "uvvis"):
            means, sds = [], []
            for task in ("Shared14", "UVSPEC15"):
                record = task_rows["modality_mean_sd"][(task, "fixed_0.5", modality, metric)]
                means.append(record["mean"])
                sds.append(record["sd"] or 0.0)
            series.append({"label": modality.upper(), "values": means, "errors": sds, "color": COLORS[modality]})
        draw_grouped_bars(canvas, box, ["Shared14", "UVSPEC15"], series, title,
                          ylabel="F1 score", ymin=0.0, ymax=1.0, legend=True)
    canvas.save(path)


def build_map_auroc_bars(path: Path, task_rows: dict) -> None:
    canvas = Canvas(1420, 720)
    canvas.text((710, 24), "B4.2 Test ranking quality by modality (threshold-free; mean ± SD, n=3 seeds)", size=21, bold=True, anchor="ma")
    specs = [
        ("macro_ap", "mAP", (110, 150, 670, 620)),
        ("macro_auroc", "Macro-AUROC", (800, 150, 1360, 620)),
    ]
    for metric, title, box in specs:
        series = []
        for modality in ("ir", "raman", "uvvis"):
            means, sds = [], []
            for task in ("Shared14", "UVSPEC15"):
                record = task_rows["modality_mean_sd"][(task, "fixed_0.5", modality, metric)]
                means.append(record["mean"])
                sds.append(record["sd"] or 0.0)
            series.append({"label": modality.upper(), "values": means, "errors": sds, "color": COLORS[modality]})
        draw_grouped_bars(canvas, box, ["Shared14", "UVSPEC15"], series, title,
                          ylabel="Score", ymin=0.0, ymax=1.0, legend=True)
    canvas.save(path)


def build_label_heatmap(path: Path, task_rows: dict, labels_by_task: dict, kind: str) -> None:
    canvas = Canvas(1740, 760)
    if kind == "f1":
        title = "B4.2 per-label Test F1 (frozen Validation-F1 thresholds; seed mean)"
        palette = "viridis"
    else:
        title = "B4.2 per-label Test AP (threshold-free; seed mean)"
        palette = "magma"
    canvas.text((870, 24), title, size=21, bold=True, anchor="ma")
    panels = [
        ("Shared14", (120, 120, 800, 480), (810, 120, 834, 480)),
        ("UVSPEC15", (960, 120, 1660, 480), (1670, 120, 1694, 480)),
    ]
    for task, box, cbar_box in panels:
        labels = labels_by_task[task]
        matrix = np.zeros((3, len(labels)))
        for row, modality in enumerate(("ir", "raman", "uvvis")):
            for col, label in enumerate(labels):
                if kind == "f1":
                    matrix[row, col] = task_rows["per_label_mean_sd"][(task, "validation_f1", modality, label)]["f1_mean"]
                else:
                    matrix[row, col] = task_rows["per_label_ap_mean_sd"][(task, modality, label)]["average_precision_mean"]
        draw_heatmap(
            canvas,
            box,
            matrix,
            ["IR", "Raman", "UV-Vis"],
            labels,
            palette=palette,
            vmin=0.0,
            vmax=1.0,
            annotate=True,
            annotate_size=9,
            row_label_size=13,
            col_label_size=9,
        )
        draw_colorbar(canvas, cbar_box, 0.0, 1.0, palette)
        canvas.text(((box[0] + box[2]) / 2, box[3] + 150), task, size=15, bold=True, anchor="ma")
    canvas.save(path)


def build_seed_stability(path: Path, task_rows: dict) -> None:
    canvas = Canvas(1420, 1060)
    canvas.text((710, 24), "B4.2 seed stability (dots = individual seeds; black = mean ± SD)", size=21, bold=True, anchor="ma")
    legend_items = [("Seed 42", COLORS["seeds"][0]), ("Seed 123", COLORS["seeds"][1]), ("Seed 2026", COLORS["seeds"][2])]
    legend_x = 560
    for label, color in legend_items:
        canvas.rect((legend_x, 62, legend_x + 14, 76), fill=color, outline=(60, 60, 60))
        canvas.text((legend_x + 20, 69), label, size=12, anchor="lm")
        legend_x += 120
    panels = [
        ("macro_f1", "validation_f1", "Macro-F1 (frozen thresholds)", 140),
        ("macro_ap", "fixed_0.5", "mAP", 620),
    ]
    tasks = ["Shared14", "UVSPEC15"]
    for metric, policy, panel_title, y_base in panels:
        for col, task in enumerate(tasks):
            x_left = 180 if col == 0 else 810
            box = (x_left, y_base, x_left + 480, y_base + 340)
            seed_values, means, sds = [], [], []
            for modality in ("ir", "raman", "uvvis"):
                seeds = task_rows["seed_values"][(task, policy, modality, metric)]
                seed_values.append([item["value"] for item in seeds])
                record = task_rows["modality_mean_sd"][(task, policy, modality, metric)]
                means.append(record["mean"])
                sds.append(record["sd"] or 0.0)
            all_values = [value for values in seed_values for value in values]
            low = min(all_values)
            high = max(all_values)
            padding = max(0.01, (high - low) * 0.25)
            draw_seed_panel(
                canvas,
                box,
                ["IR", "Raman", "UV-Vis"],
                seed_values,
                means,
                sds,
                f"{task}: {panel_title}",
                max(0.0, low - padding),
                min(1.0, high + padding),
            )
            canvas.vertical_label(box[0] - 62, (box[1] + box[3]) / 2, "Test score", size=13)
    canvas.save(path)


def build_task_comparison(path: Path, task_rows: dict) -> None:
    canvas = Canvas(1320, 700)
    canvas.text((660, 24), "B4.2 Shared14 vs UVSPEC15 by modality (mean ± SD, n=3 seeds)", size=21, bold=True, anchor="ma")
    specs = [
        ("macro_f1", "Macro-F1 (threshold 0.5)", (110, 150, 620, 600)),
        ("macro_ap", "mAP (threshold-free)", (760, 150, 1270, 600)),
    ]
    for metric, title, box in specs:
        series = []
        for task in ("Shared14", "UVSPEC15"):
            means, sds = [], []
            for modality in ("ir", "raman", "uvvis"):
                record = task_rows["modality_mean_sd"][(task, "fixed_0.5", modality, metric)]
                means.append(record["mean"])
                sds.append(record["sd"] or 0.0)
            series.append({"label": task, "values": means, "errors": sds, "color": COLORS[task]})
        draw_grouped_bars(canvas, box, ["IR", "Raman", "UV-Vis"], series, title,
                          ylabel="Score", ymin=0.0, ymax=1.0, legend=True)
    canvas.save(path)
