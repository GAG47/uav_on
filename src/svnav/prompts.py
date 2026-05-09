SVNAV_TASK1_SYSTEM_PROMPT = """
You are the Task1 semantic value reasoner for an open-world UAV visual-language navigation system.

Your job is to estimate the SEARCH VALUE of each UAV keyframe region for finding the requested target object.

Important:
You are NOT judging only whether the exact target object is already clearly visible.
You are judging whether this visible region is worth searching next.

A region can have high semantic value even if the target is not directly visible, as long as the scene context strongly suggests that the target may be nearby.

For example:
- If the target is a vehicle, roads, parking areas, driveways, garages, open streets, and vehicle-like silhouettes are relevant.
- If the target is a trash bin, sidewalks, roadsides, building entrances, courtyards, alleys, and waste collection areas are relevant.
- If the target is a food can or small object, storage areas, tables, shelves, cluttered ground, trash areas, and human activity areas may be relevant.
- If the target is a box or wooden box, storage corners, construction areas, courtyards, loading areas, warehouses, wallsides, and object piles may be relevant.

Use the full score range. Do not be overly conservative.
Do not give every frame a very low score just because the exact object is not clearly visible.
Make the scores discriminative across frames.

Semantic value calibration:
- 0.90 to 1.00: the target itself is clearly visible or almost certainly present.
- 0.70 to 0.90: the region has very strong target-related context; it is one of the best places to search.
- 0.50 to 0.70: the region is reasonably promising and should be prioritized over ordinary exploration.
- 0.30 to 0.50: weak but meaningful target-related context.
- 0.10 to 0.30: mostly irrelevant, but not impossible.
- 0.00 to 0.10: clearly irrelevant or visually uninformative.

Confidence calibration:
- High confidence if the image is clear and the relevant visual context is easy to judge.
- Medium confidence if the context is partially visible or ambiguous.
- Low confidence if the image is blurry, dark, occluded, or lacks enough scene information.

You must score every frame_id independently.
You may compare the frames only to make the scores more discriminative, but you must still return one independent score for every frame_id.

You must not output a navigation action.
You must not output forward, left, right, up, down, rotate, or stop.
You must not decide whether the UAV should stop.
You must not verify a detection box.
You must not claim the target is found unless it is visually supported.

Return only valid JSON. Do not return markdown. Do not return prose outside JSON.
""".strip()


def build_task1_user_prompt(target_text, frame_items):
    """
    Build the user prompt for SVNav Task1.

    frame_items should be a list of dicts with:
        frame_id
        step_id
        view_id
        pose_text
    """
    lines = []
    lines.append("Task: estimate region-level semantic search value for UAV navigation.")
    lines.append("")
    lines.append("Target object information:")
    lines.append(target_text if target_text else "No target description provided.")
    lines.append("")
    lines.append("You will receive multiple independent UAV keyframes.")
    lines.append("Each keyframe is labeled by frame_id, step_id, view_id, and UAV pose.")
    lines.append("")
    lines.append("Your goal:")
    lines.append("For each keyframe, estimate how useful that visible region is for finding the target object.")
    lines.append("")
    lines.append("Very important scoring rule:")
    lines.append("Do not score only based on whether the exact target object is already visible.")
    lines.append("Score based on whether the region is semantically promising for searching the target.")
    lines.append("Use different scores when frames show different levels of target-related context.")
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
    lines.append("- 0.70-0.90: very strong target-related context; top-priority search region.")
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
        '"confidence": 0.0, "reason": "short visual-grounded reason"}'
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
    lines.append("- The reason should explain the visual context, not an action.")
    lines.append("- Do not output any navigation action.")
    lines.append("- Do not output stop.")
    lines.append("- Do not use markdown or code fences.")

    return "\n".join(lines)


__all__ = [
    "SVNAV_TASK1_SYSTEM_PROMPT",
    "build_task1_user_prompt",
]


def build_task2_batch_prompt(target_info, candidates):
    """
    Build Task2 verification prompt.

    Task2 verifies whether each candidate crop corresponds to the target object.
    It must not output navigation action, stop decision, or approach decision.
    """
    target_text = target_info.text if hasattr(target_info, "text") else str(target_info)

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

Target object:
{target_text}

You will receive, for each candidate, a context image and a cropped candidate image.
The context image shows where the candidate appears in the scene.
The crop image shows the exact candidate region proposed by GroundingDINO.

Your task:
For each candidate_id, decide whether the CROP corresponds to the target object.

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
      "reason": "brief reason",
      "matched_attributes": ["short visual attributes that match"],
      "failed_attributes": ["short visual attributes that do not match or are unclear"]
    }}
  ]
}}
""".format(
        target_text=target_text,
        candidate_text="\n".join(candidate_lines),
    ).strip()

