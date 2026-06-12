import json
from flask import request, jsonify, render_template
from utils import build_image_blocks, get_context
import constants


def index():
    return render_template("index.html", styles=constants.STYLES)


def generate(client):
    images = request.files.getlist("images[]") or []
    single = request.files.get("image")
    if single and not images:
        images = [single]
    if not images:
        return jsonify({"error": "No image provided"}), 400

    images = images[:4]
    is_carousel = len(images) > 1
    carousel_ctx = "\nThis is a carousel post with multiple photos. Reference the collection as a whole." if is_carousel else ""

    ctx = get_context(request.form, constants)
    hashtags_only    = request.form.get("hashtags_only", "false").lower() == "true"
    story_mode       = request.form.get("story_mode", "false").lower() == "true"
    regenerate_index = request.form.get("regenerate_index", None)

    photo_label = "these photos" if is_carousel else "this photo"

    try:
        blocks = build_image_blocks(images)

        # ── hashtags only ──
        if hashtags_only:
            blocks.append({"type": "text", "text": f"""Look at {photo_label} and generate 20-25 relevant Instagram hashtags.
{ctx['lang_instr']}
Style: {ctx['style_desc']}{ctx['custom_context']}{ctx['audience_context']}{carousel_ctx}
Rules: Start each with #, mix popular and niche, put all on one line, output hashtags only."""})
            r = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": blocks}], max_tokens=300)
            return jsonify({"hashtags": r.choices[0].message.content.strip()})

        # ── story mode ──
        elif story_mode:
            blocks.append({"type": "text", "text": f"""Look at {photo_label} and create 3 Instagram Story captions.
Style: {ctx['style_desc']}
{ctx['lang_instr']}{ctx['mood_context']}{ctx['audience_context']}{ctx['custom_context']}
Format each story set exactly like this (separate with blank line):
Caption: [max 8 words]
POLL: [Yes/No or A/B question]
QUESTION: [open-ended question]
Output only the 3 story sets."""})
            r = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": blocks}], max_tokens=400)
            raw = r.choices[0].message.content.strip()
            stories = []
            for block in raw.split("\n\n"):
                lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
                story = {"caption": "", "poll": "", "question": ""}
                for line in lines:
                    if line.lower().startswith("caption:"):   story["caption"]  = line[8:].strip()
                    elif line.lower().startswith("poll:"):    story["poll"]     = line[5:].strip()
                    elif line.lower().startswith("question:"): story["question"] = line[9:].strip()
                if story["caption"]:
                    stories.append(story)
            stories = stories[:3]
            while len(stories) < 3:
                stories.append({"caption": "✨ Swipe up!", "poll": "Love it or nah?", "question": "What do you think?"})
            return jsonify({"stories": stories, "style": ctx["style"]["label"], "language": ctx["language"]})

        # ── regenerate one caption ──
        elif regenerate_index is not None:
            try:
                regen_idx = int(regenerate_index)
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid regenerate_index value"}), 400
            blocks.append({"type": "text", "text": f"""Look at {photo_label} and write exactly 1 Instagram caption.
Style: {ctx['style_desc']}
{ctx['length_instr']}
{ctx['lang_instr']}{ctx['mood_context']}{ctx['audience_context']}{ctx['custom_context']}{carousel_ctx}
No hashtags. Output only the caption."""})
            r = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": blocks}], max_tokens=200)
            return jsonify({"caption": r.choices[0].message.content.strip(), "index": regen_idx})

        # ── default: 3 captions ──
        else:
            carousel_note = "\n- Reference the collection/journey across multiple photos" if is_carousel else ""
            blocks.append({"type": "text", "text": f"""Look at {photo_label} and write exactly 3 different Instagram captions.
Style: {ctx['style_desc']}
{ctx['length_instr']}
{ctx['lang_instr']}{ctx['mood_context']}{ctx['audience_context']}{ctx['custom_context']}{carousel_ctx}
Rules: Exactly 3, separated by blank lines, no numbering, no hashtags, each feels different.{carousel_note}
Output only the 3 captions."""})
            r = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": blocks}], max_tokens=600)
            raw = r.choices[0].message.content.strip()
            captions = [c.strip() for c in raw.split("\n\n") if c.strip()][:3]
            while len(captions) < 3:
                captions.append(captions[-1] if captions else "✨ Beautiful moment captured.")
            return jsonify({"captions": captions, "style": ctx["style"]["label"],
                            "style2": ctx["style2"]["label"] if ctx["style2"] else None,
                            "language": ctx["language"], "length": ctx["length"],
                            "is_carousel": is_carousel, "photo_count": len(images)})

    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


def generate_bio(client):
    ctx = get_context(request.form, constants)
    custom = (request.form.get("custom_prompt") or "").strip()
    custom_context = f"\nExtra details: {custom}" if custom else ""
    try:
        r = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": f"""Generate exactly 3 different Instagram bios.
Style: {ctx['style_desc']}
{ctx['lang_instr']}{ctx['audience_context']}{ctx['mood_context']}{custom_context}
Rules: Under 150 chars each, 1-3 emojis, separated by blank lines, no numbering, output only the 3 bios."""}],
            max_tokens=300)
        raw = r.choices[0].message.content.strip()
        bios = [b.strip() for b in raw.split("\n\n") if b.strip()][:3]
        while len(bios) < 3:
            bios.append("✨ Living my best life | Creating every day")
        return jsonify({"bios": bios, "style": ctx["style"]["label"]})
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


def analyze(client):
    from utils import encode_image, get_mime
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image provided"}), 400
    try:
        img_data = encode_image(image)
        mime = get_mime(image.filename)
        r = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}},
                {"type": "text", "text": """Return a JSON object with these fields only:
{"language": "english/spanish/etc or null", "emotion": "happy/sad/excited/calm/romantic/mysterious/bold/chill/grateful/energetic/nostalgic",
"emotion_confidence": "high/medium/low", "tags": ["up to 5 tag suggestions without @"], "image_description": "one short sentence"}
No markdown, no backticks, pure JSON only."""}]}],
            max_tokens=300)
        raw = r.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        try:
            return jsonify(json.loads(raw))
        except json.JSONDecodeError:
            return jsonify({"error": "Model returned invalid JSON. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


def ab_test(client):
    images = request.files.getlist("images[]") or []
    single = request.files.get("image")
    if single and not images:
        images = [single]
    if not images:
        return jsonify({"error": "No image provided"}), 400
    images = images[:4]
    ctx = get_context(request.form, constants)
    try:
        blocks = build_image_blocks(images)
        blocks.append({"type": "text", "text": f"""Write 2 very different Instagram captions for A/B testing.
{ctx['lang_instr']}
Style: {ctx['style_desc']}{ctx['audience_context']}{ctx['mood_context']}{ctx['custom_context']}
VERSION_A: SHORT and punchy (1 sentence, under 80 chars)
VERSION_B: LONG and storytelling (3-4 sentences, emotional)
Output only these two lines."""})
        r = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": blocks}], max_tokens=400)
        raw = r.choices[0].message.content.strip()
        version_a, version_b = "", ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("VERSION_A:"): version_a = line[10:].strip()
            elif line.upper().startswith("VERSION_B:"): version_b = line[10:].strip()
        if not version_a: version_a = "✨ Moment captured."
        if not version_b: version_b = "Some moments are too beautiful to describe — they just need to be felt. ✨"
        return jsonify({"version_a": version_a, "version_b": version_b, "style": ctx["style"]["label"]})
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500
