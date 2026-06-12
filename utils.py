import base64


def encode_image(image_file):
    image_file.stream.seek(0)
    return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime(filename):
    if not filename or "." not in filename:
        return "image/jpeg"
    ext = filename.rsplit(".", 1)[-1].lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")


def build_image_blocks(images):
    blocks = []
    for img in images:
        img_data = encode_image(img)
        mime = get_mime(img.filename)
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}})
    return blocks


def get_context(form, constants):
    style_key  = form.get("style", "casual")
    style_key2 = form.get("style2", None)
    language   = form.get("language", "english")
    length     = form.get("length", "medium")
    mood       = form.get("mood", "none")
    audience   = form.get("audience", "general")
    custom     = (form.get("custom_prompt") or "").strip()

    style  = constants.STYLES.get(style_key, constants.STYLES["casual"])
    style2 = constants.STYLES.get(style_key2) if style_key2 and style_key2 in constants.STYLES else None

    mood_instr = constants.MOOD_INSTRUCTIONS.get(mood, "")

    return {
        "style":            style,
        "style2":           style2,
        "style_desc":       style["description"] if not style2 else f"{style['description']} combined with {style2['description']}",
        "length_instr":     constants.LENGTH_INSTRUCTIONS.get(length, constants.LENGTH_INSTRUCTIONS["medium"]),
        "lang_instr":       constants.LANGUAGE_INSTRUCTIONS.get(language, constants.LANGUAGE_INSTRUCTIONS["english"]),
        "mood_context":     f"\n{mood_instr}" if mood_instr else "",
        "audience_context": f"\n{constants.AUDIENCE_INSTRUCTIONS.get(audience, '')}" if constants.AUDIENCE_INSTRUCTIONS.get(audience) else "",
        "custom_context":   f"\nExtra context: {custom}" if custom else "",
        "language":         language,
        "length":           length,
    }
