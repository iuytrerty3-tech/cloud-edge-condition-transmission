from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path):
    text = path.read_bytes().decode("utf-8", errors="ignore").lstrip("\ufeff")
    return json.loads(text)


def load_json_if_exists(path: Path):
    if path.exists():
        return load_json(path)
    return None


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((width, height))
    panel = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = (width - copy.width) // 2
    y = (height - copy.height) // 2
    panel.paste(copy, (x, y))
    return panel


def draw_multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font, fill, line_gap: int) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_gap
    return y


def build_overview(dataset_root: Path) -> Path:
    design_root = dataset_root / "design_subset"
    high_root = dataset_root / "high_confidence_pattern_subset"
    candidate_root = dataset_root / "parade_pattern_candidates"

    cleaned_summary = load_json(dataset_root / "dataset_summary.json")
    paper_main_summary_path = dataset_root / "paper_main_summary.json"
    paper_main_summary = load_json(paper_main_summary_path) if paper_main_summary_path.exists() else None
    design_summary = load_json_if_exists(design_root / "summary.json")
    high_summary = load_json_if_exists(high_root / "summary.json")
    candidate_summary = load_json_if_exists(candidate_root / "summary.json")

    panels = []
    if design_summary and (design_root / "focused_montage.jpg").exists():
        panels.append(
            {
                "title": "A. Original Cultural Design Works",
                "subtitle": f"{design_summary['focused_originals']} originals across 3 categories",
                "stats": [
                    f"Porcelain: {design_summary['category_counts']['blue_and_white_porcelain']}",
                    f"Embroidery: {design_summary['category_counts']['gu_embroidery']}",
                    f"Paper-cutting: {design_summary['category_counts']['paper_cutting']}",
                ],
                "image": Image.open(design_root / "focused_montage.jpg"),
            }
        )
    if design_summary and (design_root / "patch_montage.jpg").exists():
        panels.append(
            {
                "title": "B. Patch-Level Design Subset",
                "subtitle": f"{design_summary['patches']} motif patches from design works",
                "stats": [
                    "Patch size follows local square crops",
                    "Preserves traceable source linkage",
                    "Targets visible decorative regions",
                ],
                "image": Image.open(design_root / "patch_montage.jpg"),
            }
        )
    if high_summary and candidate_summary and (high_root / "patch_montage.jpg").exists():
        panels.append(
            {
                "title": "C. High-Confidence Pattern Subset",
                "subtitle": f"{high_summary['selected_patches']} strict patches from {high_summary['unique_parents']} parade sources",
                "stats": [
                    f"Score threshold: {high_summary['score_threshold']}",
                    f"Score range: {high_summary['actual_min_score']:.2f}-{high_summary['actual_max_score']:.2f}",
                    f"Candidate pool: {candidate_summary['candidate_patches']} patches",
                ],
                "image": Image.open(high_root / "patch_montage.jpg"),
            }
        )

    if not panels:
        width = 1500
        height = 520
        canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        title_font = load_font(28)
        body_font = load_font(20)
        small_font = load_font(18)
        draw.text((50, 28), "Dataset Overview for Cultural Design Pattern Experiments", font=title_font, fill=(20, 20, 20))
        draw.text((50, 78), "The repository currently keeps only the clean main dataset categories.", font=body_font, fill=(70, 70, 70))
        lines = [
            f"Cleaned starter dataset: {cleaned_summary['total_records']}",
            (
                f"Paper-facing clean subset: {paper_main_summary['paper_main_records']}"
                if paper_main_summary
                else "Paper-facing clean subset: N/A"
            ),
            f"Blue-and-white porcelain: {cleaned_summary['category_counts'].get('blue_and_white_porcelain', 0)}",
            f"Artifact objects: {cleaned_summary['category_counts'].get('artifact_object', 0)}",
            f"Artifact patterns: {cleaned_summary['category_counts'].get('artifact_pattern', 0)}",
            f"Paper cutting: {cleaned_summary['category_counts'].get('paper_cutting', 0)}",
            f"Window flower: {cleaned_summary['category_counts'].get('window_flower', 0)}",
            f"Cultural clothing: {cleaned_summary['category_counts'].get('cultural_clothing', 0)}",
        ]
        draw_multiline(draw, (50, 130), lines, font=small_font, fill=(80, 80, 80), line_gap=10)
        out_path = dataset_root / "paper_dataset_overview.jpg"
        canvas.save(out_path, quality=93)
        return out_path

    margin = 50
    gutter = 28
    panel_width = 580
    panel_image_height = 580
    panel_text_height = 165
    title_height = 90
    footer_height = 90
    width = margin * 2 + panel_width * len(panels) + gutter * max(0, len(panels) - 1)
    height = title_height + margin + panel_image_height + panel_text_height + footer_height

    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    section_font = load_font(22)
    body_font = load_font(18)
    small_font = load_font(16)

    draw.text((margin, 24), "Dataset Overview for Cultural Design Pattern Experiments", font=title_font, fill=(20, 20, 20))
    draw.text(
        (margin, 58),
        "The figure organizes original design works, patch-level motif crops, and strict high-confidence local patterns.",
        font=small_font,
        fill=(80, 80, 80),
    )

    base_y = title_height
    for idx, panel in enumerate(panels):
        x0 = margin + idx * (panel_width + gutter)
        panel_img = fit_panel(panel["image"], panel_width, panel_image_height)
        canvas.paste(panel_img, (x0, base_y))
        draw.rectangle((x0, base_y, x0 + panel_width, base_y + panel_image_height), outline=(190, 190, 190), width=2)
        text_y = base_y + panel_image_height + 16
        draw.text((x0, text_y), panel["title"], font=section_font, fill=(20, 20, 20))
        draw.text((x0, text_y + 34), panel["subtitle"], font=body_font, fill=(60, 60, 60))
        draw_multiline(draw, (x0, text_y + 66), panel["stats"], font=small_font, fill=(80, 80, 80), line_gap=6)
        panel["image"].close()

    if paper_main_summary:
        footer_lines = [
            f"Starter dataset total: {cleaned_summary['total_records']} cleaned images; paper-facing clean subset: {paper_main_summary['paper_main_records']} images.",
        ]
        if candidate_summary:
            footer_lines.append(
                f"Supplementary parade candidate pool: {candidate_summary['candidate_patches']} patches from {paper_main_summary['supplementary_categories'].get('heritage_parade', 0)} scene-rich images."
            )
    else:
        footer_lines = [
            f"Starter dataset total: {cleaned_summary['total_records']} cleaned images.",
            "Recommended manuscript usage: original works as main dataset evidence; high-confidence patches as local motif subset.",
        ]
        if candidate_summary:
            footer_lines[0] = (
                f"Starter dataset total: {cleaned_summary['total_records']} cleaned images. Parade candidate pool: {candidate_summary['candidate_patches']} patches."
            )
    draw_multiline(draw, (margin, height - footer_height + 10), footer_lines, font=small_font, fill=(70, 70, 70), line_gap=8)

    out_path = dataset_root / "paper_dataset_overview.jpg"
    canvas.save(out_path, quality=93)
    return out_path


def write_dataset_note(dataset_root: Path) -> tuple[Path, Path]:
    cleaned_summary = load_json(dataset_root / "dataset_summary.json")
    counts = cleaned_summary["category_counts"]
    paper_main_summary_path = dataset_root / "paper_main_summary.json"
    paper_main_summary = load_json(paper_main_summary_path) if paper_main_summary_path.exists() else None
    supplementary_total = counts.get("heritage_parade", 0) + counts.get("gu_embroidery", 0)

    md_lines = [
        "# Dataset Note",
        "",
        "## Summary",
        "",
        "| Split | Count | Description |",
        "| --- | ---: | --- |",
        f"| Cleaned starter dataset | {cleaned_summary['total_records']} | All cleaned starter images currently stored in the repository |",
        (
            f"| Paper-facing clean subset | {paper_main_summary['paper_main_records']} | Clean-background museum-grade and flat paper-cut samples used as the main manuscript dataset |"
            if paper_main_summary
            else "| Paper-facing clean subset | N/A | Not generated |"
        ),
        "",
        "## Current Repository Distribution",
        "",
        f"- Blue-and-white porcelain: {counts.get('blue_and_white_porcelain', 0)}",
        f"- Artifact objects: {counts.get('artifact_object', 0)}",
        f"- Artifact patterns: {counts.get('artifact_pattern', 0)}",
        f"- Paper cutting: {counts.get('paper_cutting', 0)}",
        f"- Window flower: {counts.get('window_flower', 0)}",
        f"- Cultural clothing: {counts.get('cultural_clothing', 0)}",
        f"- Supplementary heritage parade: {counts.get('heritage_parade', 0)}",
        f"- Supplementary Gu embroidery: {counts.get('gu_embroidery', 0)}",
        "",
        "## Paper-Facing Clean Subset",
        "",
        (
            f"- Main subset size: {paper_main_summary['paper_main_records']}" if paper_main_summary else "- Main subset size: N/A"
        ),
        (
            f"- Blue-and-white porcelain: {paper_main_summary['paper_main_category_counts'].get('blue_and_white_porcelain', 0)}"
            if paper_main_summary
            else "- Blue-and-white porcelain: N/A"
        ),
        (
            f"- Artifact objects: {paper_main_summary['paper_main_category_counts'].get('artifact_object', 0)}"
            if paper_main_summary
            else "- Artifact objects: N/A"
        ),
        (
            f"- Artifact patterns: {paper_main_summary['paper_main_category_counts'].get('artifact_pattern', 0)}"
            if paper_main_summary
            else "- Artifact patterns: N/A"
        ),
        (
            f"- Paper cutting: {paper_main_summary['paper_main_category_counts'].get('paper_cutting', 0)}"
            if paper_main_summary
            else "- Paper cutting: N/A"
        ),
        (
            f"- Window flower: {paper_main_summary['paper_main_category_counts'].get('window_flower', 0)}"
            if paper_main_summary
            else "- Window flower: N/A"
        ),
        (
            f"- Cultural clothing: {paper_main_summary['paper_main_category_counts'].get('cultural_clothing', 0)}"
            if paper_main_summary
            else "- Cultural clothing: N/A"
        ),
        "",
        "## Notes",
        "",
        "- The manuscript should prioritize clean-background museum-grade porcelain, artifact, paper-cutting, window-flower, and cultural clothing samples as the main dataset evidence.",
        (
            "- Scene-rich parade images and the single embroidery sample remain supplementary assets rather than the main cultural design dataset."
            if supplementary_total
            else "- The repository currently retains only the clean main dataset categories and no supplementary scene-rich image branch."
        ),
        "- All samples remain traceable to local metadata records and source URLs.",
        "",
    ]
    md_path = dataset_root / "paper_dataset_note.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_path = dataset_root / "paper_dataset_note.json"
    json_path.write_text(
        json.dumps(
            {
                "cleaned_starter_dataset": cleaned_summary["total_records"],
                "paper_main_records": paper_main_summary["paper_main_records"] if paper_main_summary else None,
                "paper_main_category_counts": paper_main_summary["paper_main_category_counts"] if paper_main_summary else {},
                "supplementary_categories": paper_main_summary["supplementary_categories"] if paper_main_summary else {},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready dataset overview assets")
    parser.add_argument("--dataset-root", default="datasets/starter_cultural_patterns")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    overview = build_overview(dataset_root)
    md_path, json_path = write_dataset_note(dataset_root)
    print(
        json.dumps(
            {
                "overview": str(overview.resolve()).replace("\\", "/"),
                "note_md": str(md_path.resolve()).replace("\\", "/"),
                "note_json": str(json_path.resolve()).replace("\\", "/"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
