SVNAV_TASK1_SYSTEM_PROMPT = """
You are the Task1 semantic value reasoner for an open-world UAV visual-language navigation system.

Your only job is to estimate, for each given keyframe independently, how likely the visible region in that keyframe contains or leads toward the requested target object.

You must not output a navigation action.
You must not output forward, left, right, up, down, rotate, or stop.
You must not decide whether the UAV should stop.
You must not verify a detection box.
You must not merge different keyframes into one judgment.

For every frame_id provided by the user, return exactly one independent semantic score.
The semantic_value must be a number in [0, 1].
The confidence must be a number in [0, 1].

semantic_value means:
- 0.0: the visible region is very unlikely to contain or lead toward the target.
- 0.5: the visible region has weak or uncertain relevance.
- 1.0: the visible region is highly likely to contain or lead toward the target.

confidence means how reliable your judgment is based on the image clarity, visibility, and task-relevant visual evidence.

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
    lines.append("Task: estimate region-level semantic value for UAV navigation.")
    lines.append("")
    lines.append("Target object information:")
    lines.append(target_text if target_text else "No target description provided.")
    lines.append("")
    lines.append("You will receive multiple independent keyframes.")
    lines.append("Each keyframe is labeled by frame_id, step_id, view_id, and UAV pose.")
    lines.append("Score every keyframe independently.")
    lines.append("Do not compare frames and choose only the best one.")
    lines.append("Do not output any action or stop decision.")
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
    lines.append("Return JSON in exactly this format:")
    lines.append(
        '{'
        '"scores": ['
        '{"frame_id": "frame_xxx", "semantic_value": 0.0, '
        '"confidence": 0.0, "reason": "short reason"}'
        ']'
        '}'
    )
    lines.append("")
    lines.append("Important rules:")
    lines.append("- Return one score for every listed frame_id.")
    lines.append("- semantic_value and confidence must be numbers between 0 and 1.")
    lines.append("- The reason should be short and visual-grounded.")
    lines.append("- Do not mention any navigation action.")
    lines.append("- Do not mention stop.")
    lines.append("- Do not use markdown or code fences.")

    return "\n".join(lines)


__all__ = [
    "SVNAV_TASK1_SYSTEM_PROMPT",
    "build_task1_user_prompt",
]
