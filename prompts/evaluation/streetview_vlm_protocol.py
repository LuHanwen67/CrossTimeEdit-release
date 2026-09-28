"""Canonical VLM protocol shared by online GRPO rewards and offline evaluation."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


PAIR_W, PAIR_H = 1024, 512
GT_PANEL_W, GT_PANEL_H = 1024, 512


def _font(size: int):
    for path in (
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_streetview_markers(image: Image.Image) -> Image.Image:
    """Draw the shared N/W/E/S references on a cylindrical panorama."""
    image = image.convert("RGB").copy()
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = _font(max(16, height // 22))
    directions = (
        (0.00, "S", 2, (120, 160, 255), False),
        (0.25, "W", 2, (220, 220, 220), False),
        (0.50, "N", 4, (255, 255, 255), True),
        (0.75, "E", 2, (220, 220, 220), False),
        (1.00, "S", 2, (120, 160, 255), False),
    )
    for ratio, label, line_width, color, solid in directions:
        x = min(int(width * ratio), width - 1)
        if solid:
            draw.line(((x, 0), (x, height)), fill=color, width=line_width)
        else:
            for y in range(0, height, 18):
                draw.line(((x, y), (x, min(y + 11, height))), fill=color, width=line_width)
        label_color = (255, 255, 80) if label == "N" else (200, 220, 255)
        for label_y, anchor in ((6, "mt"), (height - 28, "mb")):
            for dx, dy in ((-2, -2), (2, -2), (-2, 2), (2, 2)):
                draw.text((x + dx, label_y + dy), label, fill=(0, 0, 0), font=font, anchor=anchor)
            draw.text((x, label_y), label, fill=label_color, font=font, anchor=anchor)
    return image


def _label(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = _font(max(14, image.height // 28))
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, box[2] - box[0] + 14, box[3] - box[1] + 10), fill=(0, 0, 0))
    draw.text((7, 5), text, fill=(255, 255, 255), font=font)
    return image


def create_pair_comparison(input_image: Image.Image, output_image: Image.Image) -> Image.Image:
    """Vertical input/output pair used by BP and QP."""
    input_panel = _label(
        draw_streetview_markers(input_image.resize((PAIR_W, PAIR_H), Image.Resampling.LANCZOS)),
        "Input",
    )
    output_panel = _label(
        draw_streetview_markers(output_image.resize((PAIR_W, PAIR_H), Image.Resampling.LANCZOS)),
        "Model Output",
    )
    comparison = Image.new("RGB", (PAIR_W, PAIR_H * 2))
    comparison.paste(input_panel, (0, 0))
    comparison.paste(output_panel, (0, PAIR_H))
    return comparison


def create_ia_gt_comparison(
    input_image: Image.Image, gt_image: Image.Image, output_image: Image.Image
) -> Image.Image:
    """Canonical 2x2 layout with an input, target reference, and output."""
    size = (GT_PANEL_W, GT_PANEL_H)
    input_panel = _label(
        draw_streetview_markers(input_image.resize(size, Image.Resampling.LANCZOS)),
        "Input",
    )
    gt_panel = _label(
        draw_streetview_markers(gt_image.resize(size, Image.Resampling.LANCZOS)),
        "Ground Truth",
    )
    output_panel = _label(
        draw_streetview_markers(output_image.resize(size, Image.Resampling.LANCZOS)),
        "Model Output",
    )
    comparison = Image.new("RGB", (GT_PANEL_W * 2, GT_PANEL_H * 2))
    comparison.paste(input_panel, (0, 0))
    comparison.paste(gt_panel, (GT_PANEL_W, 0))
    comparison.paste(output_panel, (0, GT_PANEL_H))
    comparison.paste(gt_panel, (GT_PANEL_W, GT_PANEL_H))
    return comparison


PAIR_PREAMBLE = """You are a top-tier computer vision and map panorama expert. Your task is to act as an objective, strict judge of a street-view image editing model.

I will provide one vertically stacked comparison image and a text editing instruction. The TOP half is labeled "Input" and is the street-view image supplied to the editing model. The BOTTOM half is labeled "Model Output" and is the generated result after editing toward the target street-view state. Use the explicit TOP/BOTTOM positions. Each half is a 360-degree cylindrical panorama with explicit directional markers:
- Center = North (N): thick solid white vertical line and yellow N text.
- Left quarter = West (W): dashed gray vertical line and W text.
- Right quarter = East (E): dashed gray vertical line and E text.
- Both left and right edges = South (S): dashed blue vertical lines and S text.

The left and right edges are physically connected in 3D space and both represent South. Judge them as a continuous cylindrical boundary.

[Text Editing Instruction]
{instruction}"""


PROMPT_IA_WITH_GT = """You are a top-tier computer vision and map panorama expert. Evaluate how well an AI model performs the requested street-view edit, using the real target street-view image as the objective reference.

[Image Layout: 2x2 grid — use these positions exactly]
- TOP-LEFT: Input, the street-view image supplied to the editing model.
- TOP-RIGHT: Ground Truth, the corresponding real target street-view.
- BOTTOM-LEFT: Model Output, generated by editing the Input toward the Ground Truth state.
- BOTTOM-RIGHT: Ground Truth repeated for direct comparison with Model Output.

All panels are 360-degree cylindrical panoramas with direction markers. Center is North, the left quarter is West, the right quarter is East, and both edges are South. The left and right edges wrap around and must be interpreted as connected. Do not swap the four panel roles: compare the BOTTOM-LEFT output primarily with the TOP-RIGHT and BOTTOM-RIGHT Ground Truth panels.

[Text Editing Instruction]
{instruction}

[Your Task]
The instruction describes the requested edit. Ground Truth shows the real target state and is the visual reference for the requested regions. Evaluate whether Model Output performs the requested changes in the correct locations and directions and resembles the corresponding target content in Ground Truth. The instruction takes priority: do not require the Output to reproduce Ground Truth changes that are not requested by the instruction.

[Assessment Criteria]
1. Completeness: Were all changes requested by the instruction performed?
2. Accuracy: Do the requested edited regions match the corresponding regions and semantics in Ground Truth?
3. Directional correctness: Were changes made in the specified North, West, East, and South regions?

Judge only instruction alignment in this dimension. Do not penalize unrelated photographic style differences, minor background preservation issues, or general image quality here; those are scored separately.

[Scoring Guide: 0-10]
- 10: Every requested edit is present in the correct location and closely matches the target.
- 7-9: Most requested edits are correct, with only minor omissions or target differences.
- 4-6: The edit is partially correct, but important requested changes are missing or differ substantially from the target.
- 1-3: Very little of the requested edit is correct, or edits go in the wrong direction.
- 0: No requested edit was performed, or the result is completely wrong.

[Strict Output Format]
Return only this JSON object, with no Markdown or additional text:
{{
  "score": [Integer 0-10]
}}"""


PROMPT_BP = PAIR_PREAMBLE + """

Evaluate Model Output strictly on [Background Preservation].

Photographic style and semantic content must be distinguished. The Input may differ from real-world reference imagery because of camera hardware and capture year. Global shifts in saturation, contrast, white balance, tone, or sharpness are allowed and must not be treated as unauthorized content changes.

Assess whether areas not requested by the instruction retain the same semantic content as the Input:
- Do not penalize changes in weather, lighting, pedestrians, or vehicles.
- Do not penalize uniform global photographic style shifts.
- Moderately penalize trees or bushes that disappear, change type, or move significantly; do not penalize slight seasonal color differences.
- Heavily penalize unauthorized changes to static structures, including buildings, roads, walls, fences, and other structural additions or removals.

[Scoring Guide: 0-10]
- 0: Catastrophic leakage; large unmentioned background regions were semantically altered.
- 1-3: Severe leakage; major static structures in unmentioned areas were added, removed, or reconstructed.
- 4-6: Moderate leakage; some static structures or large vegetation regions changed without instruction.
- 7-9: Mostly preserved; only minor semantic changes are visible.
- 10: All unmentioned areas retain the same semantic content as the Input.

[Strict Output Format]
Return only this JSON object, with no Markdown or additional text:
{{
  "Background_Preservation": {{
    "score": [Integer 0-10]
  }}
}}"""


PROMPT_QP = PAIR_PREAMBLE + """

Evaluate Model Output strictly on [Image Quality and Physics].

Assess whether the output maintains high visual fidelity, plausible physical geometry, and a seamless panoramic boundary:
- Urban spatial logic and scale: Are edited objects placed plausibly, and do their size and perspective match the cylindrical projection?
- Artifacts and noise: Are there burned or black edges, mosaic noise, unnatural blur, duplicated structures, or other generation artifacts?
- Panoramic boundary integrity: Do the left and right edges connect seamlessly without tears or gaps?
- Do not evaluate lighting or illumination.
- Do not evaluate whether the requested semantic edit is correct; that is scored by Instruction Alignment.
- Judge artifacts introduced by the Model Output relative to the Input, rather than pre-existing defects in the Input.

[Scoring Guide: 0-10]
- 0: Completely broken, with severe artifacts, impossible geometry, or unusable panoramic boundaries.
- 1-3: Major defects, including obvious artifacts or substantial scale and perspective errors.
- 4-6: Noticeable quality or geometry issues, but the image remains usable.
- 7-9: Mostly clean and physically plausible, with only slight imperfections.
- 10: Flawless visual quality, spatial logic, and 360-degree edge closure.

[Strict Output Format]
Return only this JSON object, with no Markdown or additional text:
{{
  "Image_Quality_and_Physics": {{
    "score": [Integer 0-10]
  }}
}}"""
