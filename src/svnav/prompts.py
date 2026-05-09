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
