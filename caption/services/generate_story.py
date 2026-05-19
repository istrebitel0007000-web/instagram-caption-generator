from caption.services.helpers import (
    get_style_description,
    build_image_blocks,
    sanitize_prompt,
)


def generate_story(client, images, style, style2, audience, language, mood, custom_prompt):
    """Generate 3 Instagram Story caption sets with poll and question."""
    style_desc    = get_style_description(style, style2)
    mood_note     = f" The overall mood should feel {mood}." if mood and mood != "none" else ""
    extra_note    = f" Additional context: {sanitize_prompt(custom_prompt)}" if custom_prompt.strip() else ""
    audience_note = f" Target audience: {audience}." if audience and audience != "general" else ""

    image_blocks = build_image_blocks(images)

    prompt_text = (
        f"Create 3 Instagram Story caption sets for this image. "
        f"Style: {style_desc}.{mood_note}{audience_note}{extra_note} "
        f"For each story set return EXACTLY this format:\n"
        f"CAPTION: [short punchy caption max 8 words]\n"
        f"POLL: [poll question with 2 options like 'Yes 👍 / No 👎']\n"
        f"QUESTION: [engagement question for the question sticker]\n"
        f"---\n"
        f"Return all 3 sets separated by '---'. Language: {language}."
    )

    messages = [{"role": "user", "content": image_blocks + [{"type": "text", "text": prompt_text}]}]
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=600,
    )
    raw = response.choices[0].message.content.strip()

    stories = []
    for block in raw.split("---"):
        block = block.strip()
        if not block:
            continue
        cap = poll = question = ""
        for line in block.splitlines():
            if line.startswith("CAPTION:"):
                cap = line.replace("CAPTION:", "").strip()
            elif line.startswith("POLL:"):
                poll = line.replace("POLL:", "").strip()
            elif line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
        if cap:
            stories.append({"caption": cap, "poll": poll, "question": question})

    return {"stories": stories[:3]}
