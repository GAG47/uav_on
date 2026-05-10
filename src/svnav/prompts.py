from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


SVNAV_TASK1_SYSTEM_PROMPT = """
You are the Task1 semantic value reasoner for an open-world UAV visual-language navigation system.

Your job is to estimate the SEARCH VALUE of each UAV keyframe region for finding the requested target object.

Important:
- You are NOT outputting a navigation action.
- You are NOT deciding stop.
- You are NOT verifying a GDINO detection box.
- You are estimating whether the visible region is useful for searching the target.

Use the full target object profile, including:
- target name
- size
- detailed visual description
- functional context
- any instruction text

A region can have high semantic value even if the exact object is not clearly visible, as long as the scene context is likely to lead to the target.
For example:
- If the target is a vehicle, roads, parking areas, driveways, garages, open streets, and vehicle-like silhouettes are relevant.
- If the target is a trash bin, sidewalks, roadsides, building entrances, courtyards, alleys, and waste collection areas are relevant.
- If the target is a food can or small object, storage areas, tables, shelves, cluttered ground, trash areas, and human activity areas may be relevant.
- If the target is a box or wooden box, storage corners, construction areas, courtyards, loading areas, warehouses, wallsides, and object piles may be relevant.

Use detailed target attributes carefully:
- visual attributes such as material, color, shape, size, handles, reinforcements, stacked structure, or container-like form can increase search value;
- but do not claim the target is found unless the visible evidence supports it.

Semantic value calibration:
- 0.90 to 1.00: the target itself is clearly visible or almost certainly present.
- 0.70 to 0.90: the region has very strong target-related context or contains highly similar visible objects.
- 0.50 to 0.70: the region is reasonably promising and should be prioritized over ordinary exploration.
- 0.30 to 0.50: weak but meaningful target-related context.
- 0.10 to 0.30: mostly irrelevant, but not impossible.
- 0.00 to 0.10: clearly irrelevant or visually uninformative.

Confidence calibration:
- High confidence if the image is clear and relevant visual context is easy to judge.
- Medium confidence if the context is partially visible or ambiguous.
- Low confidence if the image is blurry, dark, occluded, or lacks enough scene information.

Return only valid JSON.
Do not return markdown.
Do not return prose outside JSON.
""".strip()


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_camel_case(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[-_/\\]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _get_target_field(target_info: Any, name: str, default: Optional[str] = None) -> Optional[str]:
    if target_info is None:
        return default
    if isinstance(target_info, dict):
        value = target_info.get(name, default)
    else:
        value = getattr(target_info, name, default)
    if value is None:
        return default
    return _clean_text(value)


def _target_name(target_info: Any) -> str:
    if hasattr(target_info, "name"):
        return _clean_text(getattr(target_info, "name"))
    if isinstance(target_info, dict):
        return _clean_text(target_info.get("name") or target_info.get("true_name") or target_info.get("object_name"))
    return _clean_text(target_info)


def build_target_profile_text(target_info: Any) -> str:
    """
    Build a detailed target profile for Task1 and Task2.

    This intentionally keeps the dataset description visible to the VLM.
    """
    if hasattr(target_info, "text"):
        base_text = _clean_text(target_info.text)
    else:
        base_text = ""

    name = _get_target_field(target_info, "name") or _target_name(target_info)
    size = _get_target_field(target_info, "size")
    description = _get_target_field(target_info, "description")
    instruction = _get_target_field(target_info, "instruction")
    category = _get_target_field(target_info, "category")
    object_name = _get_target_field(target_info, "object_name")
    true_name = _get_target_field(target_info, "true_name")

    lines = []
    if name:
        lines.append("Target name: {}".format(name))
    if true_name and true_name != name:
        lines.append("Dataset true name: {}".format(true_name))
    if size:
        lines.append("Target size: {}".format(size))
    if category:
        lines.append("Target category: {}".format(category))
    if description:
        lines.append("Detailed visual description: {}".format(description))
    if instruction:
        lines.append("Task instruction/context: {}".format(instruction))
    if object_name:
        lines.append("Dataset object asset name for logging only: {}".format(object_name))

    if not lines and base_text:
        lines.append(base_text)
    if not lines:
        lines.append("No detailed target description provided.")

    return "\n".join(lines)


def _wooden_box_reject_rules(target_text: str) -> List[str]:
    text = target_text.lower()
    if not (
        "woodenbox" in text
        or "wooden box" in text
        or "wood box" in text
        or "wooden crate" in text
        or "crate" in text
    ):
        return []

    return [
        "Reject roofs, huts, wooden walls, fences, planks, doors, windows, floors, terrain, and generic wooden structures unless the crop clearly shows a box/crate object.",
        "For WoodenBox-like targets, prefer crops showing a cubic or stacked wooden crate/box, weathered light-brown wood, blue metal corner reinforcements, rope handles, or a storage/transport container shape.",
        "A large wooden building surface is not a wooden box.",
    ]


def build_target_reject_rules(target_info: Any) -> List[str]:
    target_text = build_target_profile_text(target_info)
    rules = []

    rules.extend(_wooden_box_reject_rules(target_text))

    lowered = target_text.lower()
    if "foodcan" in lowered or "food can" in lowered or "can" in lowered:
        rules.extend([
            "Reject barrels, trash bins, cylinders that are too large, poles, pipes, and generic round structures unless the crop clearly shows a small food can.",
            "A food can should look like a small container, not a large industrial object.",
        ])

    if "trash" in lowered or "garbage" in lowered or "bin" in lowered:
        rules.extend([
            "Reject boxes, barrels, walls, and dark background regions unless the crop clearly shows a trash bin or waste container.",
        ])

    if not rules:
        rules.append("Reject background, terrain, roofs, walls, shadows, vegetation, and unrelated objects.")

    return rules


def build_task1_user_prompt(target_text, frame_items):
    """
    Build the user prompt for SVNav Task1.

    frame_items should be a list of dicts with:
        frame_id
        step_id
        view_id
        pose_text
    """
    target_profile = target_text if isinstance(target_text, str) else build_target_profile_text(target_text)

    lines = []
    lines.append("Task: estimate region-level semantic search value for UAV navigation.")
    lines.append("")
    lines.append("Target object profile:")
    lines.append(_clean_text(target_profile) if target_profile else "No target description provided.")
    lines.append("")
    lines.append("How to use the target profile:")
    lines.append("- Use the detailed visual description, size, material, color, shape, and function to judge search value.")
    lines.append("- Do not require the exact target to be fully visible; promising context is enough for a high search value.")
    lines.append("- Do not blindly score generic scenery high unless it visually or contextually supports the target profile.")
    lines.append("- If the target description mentions specific details such as blue metal reinforcements, rope handles, stacked crates, small cans, or storage containers, use those details in your reasoning.")
    lines.append("")
    lines.append("You will receive multiple independent UAV keyframes.")
    lines.append("Each keyframe is labeled by frame_id, step_id, view_id, and UAV pose.")
    lines.append("")
    lines.append("Your goal:")
    lines.append("For each keyframe, estimate how useful that visible region is for finding the target object.")
    lines.append("")
    lines.append("Keyframes:")
    for idx, item in enumerate(frame_items):
        lines.append(
            "{}. frame_id={}, step_id={}, view_id={}, pose={}".format(
                idx + 1,
                item.get("frame_id"),
                item.get("step_id"),
                item.get("view_id"),
                item.get("pose_text"),
            )
        )

    lines.append("")
    lines.append("Score calibration:")
    lines.append("- 0.90-1.00: target itself is clearly visible or almost certainly present.")
    lines.append("- 0.70-0.90: very strong target-related context or highly similar visible objects.")
    lines.append("- 0.50-0.70: reasonably promising region; should influence navigation.")
    lines.append("- 0.30-0.50: weak but meaningful target-related context.")
    lines.append("- 0.10-0.30: mostly irrelevant, but not impossible.")
    lines.append("- 0.00-0.10: clearly irrelevant or visually uninformative.")
    lines.append("")
    lines.append("Return JSON in exactly this format:")
    lines.append(
        '{'
        '"scores": ['
        '{"frame_id": "frame_xxx", "semantic_value": 0.0, '
        '"confidence": 0.0, "reason": "short visual-grounded reason using target attributes when relevant"}'
        ']'
        '}'
    )
    lines.append("")
    lines.append("Rules:")
    lines.append("- Return one score for every listed frame_id.")
    lines.append("- semantic_value must be a number between 0 and 1.")
    lines.append("- confidence must be a number between 0 and 1.")
    lines.append("- Use the full score range when appropriate.")
    lines.append("- Make the scores discriminative across frames.")
    lines.append("- The reason should explain the visual/context evidence, not an action.")
    lines.append("- Do not output any navigation action.")
    lines.append("- Do not output stop.")
    lines.append("- Do not use markdown or code fences.")
    return "\n".join(lines)


def build_task2_batch_prompt(target_info, candidates):
    """
    Build Task2 verification prompt.

    Task2 verifies whether each candidate crop corresponds to the target object.
    It must not output navigation action, stop decision, or approach decision.
    """
    target_profile = build_target_profile_text(target_info)
    reject_rules = build_target_reject_rules(target_info)

    candidate_lines = []
    for idx, item in enumerate(candidates, start=1):
        candidate_lines.append(
            "Candidate {idx}:\n"
            "- candidate_id: {candidate_id}\n"
            "- view: {view_id}\n"
            "- step: {step_id}\n"
            "- GDINO label: {label}\n"
            "- GDINO score: {score}\n"
            "- bbox: {bbox}\n"
            "- bbox area ratio: {area_ratio}\n"
            "- quality_score: {quality_score}\n"
            "- position_3d: {position_3d}\n"
            "- position_confidence: {position_confidence}\n"
            "- depth_valid: {depth_valid}\n"
            "- geometry_summary: {geometry_summary}\n".format(
                idx=idx,
                candidate_id=item.get("candidate_id"),
                view_id=item.get("view_id"),
                step_id=item.get("step_id"),
                label=item.get("label"),
                score=item.get("score"),
                bbox=item.get("bbox"),
                area_ratio=item.get("area_ratio"),
                quality_score=item.get("quality_score"),
                position_3d=item.get("position_3d"),
                position_confidence=item.get("position_confidence"),
                depth_valid=item.get("depth_valid"),
                geometry_summary=item.get("geometry_summary"),
            )
        )

    return """
You are verifying object candidates for an open-world UAV navigation task.

Target object profile:
{target_profile}

Important target-verification principle:
Use the detailed visual description as the main reference. Match the candidate crop against concrete target attributes such as object type, material, color, shape, size, stacked structure, handles, reinforcements, and function.

You will receive, for each candidate, a context image and a cropped candidate image.
The context image shows where the candidate appears in the scene.
The crop image shows the exact candidate region proposed by GroundingDINO.

Your task:
For each candidate_id, decide whether the CROP corresponds to the target object.

Positive evidence:
- The crop clearly shows the target object or a highly specific visual match to the target profile.
- For detailed descriptions, matched attributes should be explicitly reflected in matched_attributes.

Negative evidence:
{reject_rules}

Important rules:
- Judge only the candidate crop and its marked candidate, not other objects in the full image.
- Do not output any navigation action.
- Do not output stop.
- Do not decide whether the UAV has arrived.
- Do not use bbox size as arrival evidence.
- If the crop is unclear, partially visible, background, roof, wall, shadow, tree, terrain, or a wrong object, return "no" or "maybe".
- Return "match" only when the crop visually matches the target object well.
- Use "maybe" when it is plausible but not reliable enough.
- Use "no" when it is likely not the target.

Candidate metadata:
{candidate_text}

Return JSON only, with this schema:
{{
  "results": [
    {{
      "candidate_id": "string",
      "verdict": "match | maybe | no",
      "confidence": 0.0,
      "reason": "brief reason grounded in the target description and candidate crop",
      "matched_attributes": ["short visual attributes that match"],
      "failed_attributes": ["short visual attributes that do not match or are unclear"]
    }}
  ]
}}
""".format(
        target_profile=target_profile,
        reject_rules="\n".join("- " + rule for rule in reject_rules),
        candidate_text="\n".join(candidate_lines),
    ).strip()


# ----------------------------------------------------------------------
# GDINO detection prompt
# ----------------------------------------------------------------------

def _add_phrase(phrases: List[str], phrase: Any, max_words: int = 6) -> None:
    phrase = _split_camel_case(phrase)
    if not phrase:
        return
    words = phrase.split()
    if len(words) > max_words:
        return
    normalized = " ".join(words)
    if normalized not in phrases:
        phrases.append(normalized)


def _extract_visual_words(text: Any) -> List[str]:
    text = _split_camel_case(text)
    colors = {
        "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
        "brown", "orange", "pink", "purple", "cyan", "silver", "golden",
        "light", "dark",
    }
    materials = {
        "wood", "wooden", "metal", "metallic", "plastic", "stone", "concrete",
        "glass", "fabric", "cloth", "leather", "rope", "paper", "cardboard",
    }
    shapes = {
        "small", "medium", "large", "big", "round", "square", "rectangular",
        "cubic", "cube", "long", "short", "tall", "flat", "stacked",
        "reinforced", "corner", "corners", "handle", "handles",
    }

    selected = []
    for word in text.split():
        if word in colors or word in materials or word in shapes:
            if word not in selected:
                selected.append(word)
    return selected


def _aliases_for_target_name(name: Any) -> List[str]:
    compact = str(name or "").replace(" ", "").replace("_", "").replace("-", "").lower()
    spaced = _split_camel_case(name)
    aliases = []

    if "woodenbox" in compact or "wooden box" in spaced or "wood box" in spaced:
        aliases.extend([
            "wooden box",
            "wooden crate",
            "wood box",
            "storage box",
            "stacked wooden crates",
            "crate stack",
            "box with blue metal corners",
            "crate with rope handles",
            "box",
        ])
    elif "foodcan" in compact or "food can" in spaced:
        aliases.extend([
            "food can",
            "tin can",
            "metal can",
            "small can",
            "canned food",
            "can",
        ])
    elif "trashcan" in compact or "trash can" in spaced or "garbage bin" in spaced:
        aliases.extend([
            "trash can",
            "garbage can",
            "waste bin",
            "dustbin",
            "bin",
        ])
    elif "barrel" in compact or "barrel" in spaced:
        aliases.extend([
            "barrel",
            "oil barrel",
            "drum",
        ])
    elif "chair" in spaced:
        aliases.extend([
            "chair",
            "seat",
            "outdoor chair",
        ])
    elif "car" in spaced or "sedan" in spaced:
        aliases.extend([
            "car",
            "sedan",
            "vehicle",
        ])
    elif "tent" in spaced:
        aliases.extend([
            "tent",
            "camping tent",
        ])
    elif "boat" in spaced:
        aliases.extend([
            "boat",
            "small boat",
            "watercraft",
        ])
    else:
        aliases.append(spaced)

    words = spaced.split()
    if len(words) >= 2:
        aliases.append(" ".join(words[-2:]))
    if len(words) >= 1:
        aliases.append(words[-1])

    return aliases


def build_gdino_detection_prompt(target_info, max_phrases: int = 10, max_words: int = 6) -> str:
    """
    Build short open-vocabulary detection phrases for GroundingDINO.

    GDINO should receive compact noun phrases, not long reasoning instructions.
    Detailed target description is used here only to extract short visual phrases.
    Full description is still used by Task1 and Task2.
    """
    name = _get_target_field(target_info, "name") or _target_name(target_info)
    size = _get_target_field(target_info, "size")
    description = _get_target_field(target_info, "description")
    instruction = _get_target_field(target_info, "instruction")

    phrases: List[str] = []
    spaced_name = _split_camel_case(name)
    compact_name = str(name or "").lower().strip()

    _add_phrase(phrases, spaced_name, max_words=max_words)
    if compact_name and compact_name.replace(" ", "") != spaced_name.replace(" ", ""):
        _add_phrase(phrases, compact_name, max_words=max_words)

    for alias in _aliases_for_target_name(name):
        _add_phrase(phrases, alias, max_words=max_words)

    visual_text = " ".join(str(x or "") for x in [size, description, instruction] if x)
    visual_words = _extract_visual_words(visual_text)

    base_phrases = list(phrases[:5])
    for attr in visual_words:
        for base in base_phrases:
            if attr in base.split():
                continue
            _add_phrase(phrases, "{} {}".format(attr, base), max_words=max_words)
            if len(phrases) >= max_phrases:
                break
        if len(phrases) >= max_phrases:
            break

    if not phrases:
        _add_phrase(phrases, "object", max_words=max_words)

    return ". ".join(phrases[:max_phrases]) + "."


__all__ = [
    "SVNAV_TASK1_SYSTEM_PROMPT",
    "build_target_profile_text",
    "build_target_reject_rules",
    "build_task1_user_prompt",
    "build_task2_batch_prompt",
    "build_gdino_detection_prompt",
]
