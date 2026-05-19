import base64

STYLES = {
    "casual":       {"label": "Casual & Fun",    "emoji": "😎", "description": "relaxed, fun, and conversational tone with emojis"},
    "aesthetic":    {"label": "Aesthetic",        "emoji": "✨", "description": "dreamy, poetic, and visually descriptive tone"},
    "motivational": {"label": "Motivational",     "emoji": "💪", "description": "inspiring, energetic, and uplifting tone"},
    "funny":        {"label": "Funny & Witty",    "emoji": "😂", "description": "humorous, clever, and playful tone with jokes or puns"},
    "professional": {"label": "Professional",     "emoji": "💼", "description": "polished, formal, and business-appropriate tone"},
    "romantic":     {"label": "Romantic",         "emoji": "❤️", "description": "loving, warm, and heartfelt tone"},
}


def encode_image(file):
    """Encode an uploaded file to base64 string."""
    file.seek(0)  # Fix: reset file pointer before reading
    return base64.b64encode(file.read()).decode("utf-8")


def get_image_media_type(filename):
    """Return MIME type based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    mapping = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
        "gif":  "image/gif",
    }
    return mapping.get(ext, "image/jpeg")


def get_style_description(style, style2=None):
    """Build style description string, optionally blending two styles."""
    s1 = STYLES.get(style, STYLES["casual"])["description"]
    if style2 and style2 in STYLES:
        s2 = STYLES[style2]["description"]
        return f"{s1}, blended with {s2}"
    return s1


def build_image_blocks(images):
    """Build image content blocks for Groq API from uploaded files."""
    blocks = []
    for img in images[:4]:
        b64 = encode_image(img)
        mime = get_image_media_type(img.filename)
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
    return blocks


def parse_numbered_list(raw):
    """Parse numbered list from AI response into a list of strings."""
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if line and len(line) > 1 and line[0].isdigit() and line[1] in ".):" :
            items.append(line[2:].strip())
    if not items:
        # Fallback: split by double newline
        items = [i.strip() for i in raw.split("\n\n") if i.strip()]
    return items


def sanitize_prompt(text, max_length=500):
    """Sanitize and limit custom prompt length."""
    if not text:
        return ""
    return text.strip()[:max_length]
