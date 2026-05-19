from caption.services.helpers import (
    get_style_description,
    build_image_blocks,
    sanitize_prompt,
)


def ab_test_caption(client, images, style, style2, audience, language, mood, custom_prompt):
    """Generate A/B test captions — Version A short, Version B long."""
    style_desc = get_style_description(style, style2)
    mood_note  = f" Mood: {mood}." if mood and mood != "none" else ""
    extra_note = f" Context: {sanitize_prompt(custom_prompt)}" if custom_prompt.strip() else ""

    image_blocks = build_image_blocks(images[:1])

    prompt_text = (
        f"Write 2 Instagram captions for this image in A/B test format.\n"
        f"Style: {style_desc}.{mood_note} Audience: {audience}. Language: {language}.{extra_note}\n\n"
        f"VERSION_A: [short punchy caption, 1-2 sentences]\n"
        f"VERSION_B: [detailed engaging caption, 4-5 sentences with emojis]\n\n"
        f"Return exactly these two labels followed by the captions. Nothing else."
    )

    messages = [{"role": "user", "content": image_blocks + [{"type": "text", "text": prompt_text}]}]
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()

    version_a = version_b = ""
    for line in raw.splitlines():
        if line.startswith("VERSION_A:"):
            version_a = line.replace("VERSION_A:", "").strip()
        elif line.startswith("VERSION_B:"):
            version_b = line.replace("VERSION_B:", "").strip()

    # Fallback if parsing fails
    if not version_a or not version_b:
        parts = raw.split("\n\n")
        version_a = parts[0].strip() if len(parts) > 0 else raw
        version_b = parts[1].strip() if len(parts) > 1 else raw

    return {"version_a": version_a, "version_b": version_b}
