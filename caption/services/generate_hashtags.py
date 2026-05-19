from caption.services.helpers import (
    get_style_description,
    build_image_blocks,
    sanitize_prompt,
)


def generate_hashtags(client, images, style, audience, custom_prompt):
    """Generate 20-25 relevant Instagram hashtags for the given image."""
    style_desc    = get_style_description(style)
    extra_note    = f" Additional context: {sanitize_prompt(custom_prompt)}" if custom_prompt.strip() else ""
    audience_note = f" Target audience: {audience}." if audience and audience != "general" else ""

    image_blocks = build_image_blocks(images)

    prompt_text = (
        f"Generate 20-25 highly relevant Instagram hashtags for this image. "
        f"Style: {style_desc}.{audience_note}{extra_note} "
        f"Mix popular, niche, and branded tags. Return only the hashtags separated by spaces, nothing else."
    )

    messages = [{"role": "user", "content": image_blocks + [{"type": "text", "text": prompt_text}]}]
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=300,
    )
    return {"hashtags": response.choices[0].message.content.strip()}
