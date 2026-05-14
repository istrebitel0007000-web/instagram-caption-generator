import os
import base64
from flask import Flask, request, jsonify, render_template
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

STYLES = {
    "casual": {"label": "Casual & Fun", "emoji": "😎", "description": "relaxed, fun, and conversational tone with emojis"},
    "aesthetic": {"label": "Aesthetic", "emoji": "✨", "description": "dreamy, poetic, and visually descriptive tone"},
    "motivational": {"label": "Motivational", "emoji": "💪", "description": "inspiring, energetic, and uplifting tone"},
    "funny": {"label": "Funny & Witty", "emoji": "😂", "description": "humorous, clever, and playful tone with jokes or puns"},
    "professional": {"label": "Professional", "emoji": "💼", "description": "polished, formal, and business-appropriate tone"},
    "romantic": {"label": "Romantic", "emoji": "❤️", "description": "loving, warm, and heartfelt tone"},
}

LENGTH_INSTRUCTIONS = {
    "short": "Keep each caption very short — maximum 1-2 sentences, under 80 characters. Be punchy and impactful.",
    "medium": "Keep each caption medium length — 2-3 sentences, around 100-180 characters.",
    "long": "Write longer, detailed captions — 3-5 sentences, around 200-350 characters. Include more storytelling.",
}

LANGUAGE_INSTRUCTIONS = {
    "english": "Write in English.",
    "spanish": "Write in Spanish (Español).",
    "french": "Write in French (Français).",
    "german": "Write in German (Deutsch).",
    "portuguese": "Write in Portuguese (Português).",
    "arabic": "Write in Arabic (العربية).",
    "russian": "Write in Russian (Русский).",
    "uzbek": "Write in Uzbek (O'zbek tili).",
}

MOOD_INSTRUCTIONS = {
    "none": "",
    "happy": "Make the captions feel joyful, bright, and celebratory.",
    "sad": "Make the captions feel nostalgic, melancholic, and reflective.",
    "mysterious": "Make the captions feel mysterious, intriguing, and thought-provoking.",
    "bold": "Make the captions feel bold, powerful, confident, and strong.",
    "chill": "Make the captions feel relaxed, peaceful, and laid-back.",
    "grateful": "Make the captions feel grateful, appreciative, and warm-hearted.",
}

AUDIENCE_INSTRUCTIONS = {
    "general": "",
    "teens": "Target audience: teenagers (13-19). Use trendy slang, pop culture references, energetic tone. Keep it short and punchy.",
    "professionals": "Target audience: working professionals. Use sophisticated vocabulary, career-relevant references, polished tone.",
    "fitness": "Target audience: fitness enthusiasts. Use gym/workout terminology, motivational language, health-focused references.",
    "foodies": "Target audience: food lovers. Use mouth-watering descriptions, culinary terms, food culture references.",
    "travelers": "Target audience: travel enthusiasts. Use wanderlust language, adventure references, destination-focused vocabulary.",
    "parents": "Target audience: parents and families. Use warm, relatable family content, parenting humor, wholesome references.",
    "entrepreneurs": "Target audience: entrepreneurs and business owners. Use hustle culture language, success mindset, business references.",
    "creatives": "Target audience: artists and creative professionals. Use artistic vocabulary, creative process references, inspiration-focused language.",
}


def encode_image(image_file):
    data = image_file.read()
    return base64.b64encode(data).decode("utf-8")


def get_mime(filename):
    ext = (filename.rsplit(".", 1)[-1].lower()) if filename else "jpeg"
    return f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"


def build_image_blocks(images):
    blocks = []
    for img in images:
        img_data = encode_image(img)
        mime = get_mime(img.filename)
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img_data}"}
        })
    return blocks


@app.route("/")
def index():
    return render_template("index.html", styles=STYLES)


@app.route("/generate", methods=["POST"])
def generate():
    images = request.files.getlist("images[]")
    single_image = request.files.get("image")
    if single_image and not images:
        images = [single_image]

    style_key = request.form.get("style", "casual")
    style_key2 = request.form.get("style2", None)
    language = request.form.get("language", "english")
    length = request.form.get("length", "medium")
    mood = request.form.get("mood", "none")
    audience = request.form.get("audience", "general")
    custom_prompt = (request.form.get("custom_prompt") or "").strip()
    hashtags_only = request.form.get("hashtags_only", "false").lower() == "true"
    regenerate_index = request.form.get("regenerate_index", None)
    story_mode = request.form.get("story_mode", "false").lower() == "true"

    if not images:
        return jsonify({"error": "No image provided"}), 400

    images = images[:4]
    is_carousel = len(images) > 1

    style = STYLES.get(style_key, STYLES["casual"])
    style2 = STYLES.get(style_key2, None) if style_key2 else None
    style_desc = style["description"]
    if style2:
        style_desc = f"{style['description']} combined with {style2['description']}"

    length_instr = LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS["medium"])
    lang_instr = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["english"])
    mood_instr = MOOD_INSTRUCTIONS.get(mood, "")
    audience_instr = AUDIENCE_INSTRUCTIONS.get(audience, "")

    custom_context = f"\nExtra context: {custom_prompt}" if custom_prompt else ""
    mood_context = f"\nMood: {mood_instr}" if mood_instr else ""
    audience_context = f"\n{audience_instr}" if audience_instr else ""
    carousel_context = "\nThis is a carousel/gallery post with multiple photos. Reference the collection as a whole." if is_carousel else ""

    try:
        image_blocks = build_image_blocks(images)

        if hashtags_only:
            prompt = f"""Look at {'these photos' if is_carousel else 'this photo'} and generate 20-25 relevant Instagram hashtags.
{lang_instr}
Style context: {style_desc}{custom_context}{audience_context}{carousel_context}

Rules:
- Start each hashtag with #
- Mix popular and niche hashtags
- Make them relevant to the image content
- Put them all on one line separated by spaces
- Only output the hashtags, nothing else"""

            image_blocks.append({"type": "text", "text": prompt})
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": image_blocks}],
                max_tokens=300,
            )
            return jsonify({"hashtags": response.choices[0].message.content.strip()})

        elif story_mode:
            prompt = f"""Look at {'these photos' if is_carousel else 'this photo'} and create 3 Instagram Story captions.
Style: {style_desc}
{lang_instr}{mood_context}{audience_context}{custom_context}

Rules:
- Each story caption must be VERY SHORT — maximum 8 words
- After each caption, add a poll suggestion starting with "POLL:" and a question suggestion starting with "QUESTION:"
- Separate each story set with a blank line
- Format exactly like this:
Caption: [short caption here]
POLL: [Yes/No or A/B poll question]
QUESTION: [open-ended question to ask followers]

- Only output the 3 story sets, nothing else"""

            image_blocks.append({"type": "text", "text": prompt})
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": image_blocks}],
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            stories = []
            for block in raw.split("\n\n"):
                lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
                story = {"caption": "", "poll": "", "question": ""}
                for line in lines:
                    if line.lower().startswith("caption:"):
                        story["caption"] = line[8:].strip()
                    elif line.lower().startswith("poll:"):
                        story["poll"] = line[5:].strip()
                    elif line.lower().startswith("question:"):
                        story["question"] = line[9:].strip()
                if story["caption"]:
                    stories.append(story)
            stories = stories[:3]
            while len(stories) < 3:
                stories.append({"caption": "✨ Swipe up!", "poll": "Love it or nah?", "question": "What do you think?"})
            return jsonify({
                "stories": stories,
                "style": style["label"],
                "language": language,
            })

        elif regenerate_index is not None:
            prompt = f"""Look at {'these photos' if is_carousel else 'this photo'} and write exactly 1 Instagram caption.
Style: {style_desc}
{length_instr}
{lang_instr}{mood_context}{audience_context}{custom_context}{carousel_context}

Rules:
- Write only 1 caption
- Include relevant emojis naturally
- Do NOT include hashtags
- Only output the caption text, nothing else"""

            image_blocks.append({"type": "text", "text": prompt})
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": image_blocks}],
                max_tokens=200,
            )
            caption = response.choices[0].message.content.strip()
            return jsonify({"caption": caption, "index": int(regenerate_index)})

        else:
            carousel_note = "\n- Since this is a carousel post, reference the collection/journey/story across multiple photos" if is_carousel else ""
            prompt = f"""Look at {'these photos' if is_carousel else 'this photo'} and write exactly 3 different Instagram captions.
Style: {style_desc}
{length_instr}
{lang_instr}{mood_context}{audience_context}{custom_context}{carousel_context}

Rules:
- Write EXACTLY 3 captions
- Separate each caption with a blank line
- Do NOT number them or add labels
- Include relevant emojis naturally
- Do NOT include hashtags
- Make each caption feel different from the others{carousel_note}
- Only output the 3 captions, nothing else"""

            image_blocks.append({"type": "text", "text": prompt})
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": image_blocks}],
                max_tokens=600,
            )

            raw = response.choices[0].message.content.strip()
            captions = [c.strip() for c in raw.split("\n\n") if c.strip()]
            captions = captions[:3]
            while len(captions) < 3:
                captions.append(captions[-1] if captions else "✨ Beautiful moment captured.")

            return jsonify({
                "captions": captions,
                "style": style["label"],
                "style2": style2["label"] if style2 else None,
                "language": language,
                "length": length,
                "is_carousel": is_carousel,
                "photo_count": len(images),
            })

    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


@app.route("/bio", methods=["POST"])
def generate_bio():
    style_key = request.form.get("style", "casual")
    audience = request.form.get("audience", "general")
    language = request.form.get("language", "english")
    custom_prompt = (request.form.get("custom_prompt") or "").strip()

    style = STYLES.get(style_key, STYLES["casual"])
    lang_instr = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["english"])
    audience_instr = AUDIENCE_INSTRUCTIONS.get(audience, "")
    custom_context = f"\nExtra details: {custom_prompt}" if custom_prompt else ""

    try:
        prompt = f"""Generate exactly 3 different Instagram bio options.
Style: {style["description"]}
{lang_instr}
{audience_instr}{custom_context}

Rules:
- Each bio must be under 150 characters (Instagram limit)
- Include 1-3 relevant emojis per bio
- Make each bio feel unique and different
- Include a subtle call to action in at least one bio
- Separate each bio with a blank line
- Do NOT number them or add labels
- Only output the 3 bios, nothing else"""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()
        bios = [b.strip() for b in raw.split("\n\n") if b.strip()]
        bios = bios[:3]
        while len(bios) < 3:
            bios.append("✨ Living my best life | Creating every day")

        return jsonify({"bios": bios, "style": style["label"]})

    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """Auto-detect language, emotion, and suggest tags from image."""
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image provided"}), 400
    try:
        img_data = encode_image(image)
        mime = get_mime(image.filename)
        prompt = """Analyze this image carefully and return a JSON object with exactly these fields:
{
  "language": "detected language code if there is visible text in the image (english/spanish/french/german/portuguese/arabic/russian/uzbek), or null if no text",
  "emotion": "dominant emotion/mood detected (happy/sad/excited/calm/romantic/mysterious/bold/chill/grateful/energetic/nostalgic), pick the single best one",
  "emotion_confidence": "high/medium/low",
  "tags": ["up to 5 relevant @tag suggestions for brands, locations, or account types - just the word without @"],
  "image_description": "one short sentence describing the image content"
}

Rules:
- Only return the JSON object, nothing else
- No markdown, no backticks, just pure JSON
- For tags suggest relevant Instagram accounts/brands/locations based on what you see
- Be specific with emotion detection based on faces, colors, and overall mood"""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        # Clean up any markdown
        raw = raw.replace("```json", "").replace("```", "").strip()
        import json
        result = json.loads(raw)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


@app.route("/ab_test", methods=["POST"])
def ab_test():
    """Generate 2 very different caption versions for A/B testing."""
    images = request.files.getlist("images[]")
    single_image = request.files.get("image")
    if single_image and not images:
        images = [single_image]
    if not images:
        return jsonify({"error": "No image provided"}), 400

    images = images[:4]
    style_key = request.form.get("style", "casual")
    language = request.form.get("language", "english")
    audience = request.form.get("audience", "general")
    custom_prompt = (request.form.get("custom_prompt") or "").strip()

    style = STYLES.get(style_key, STYLES["casual"])
    lang_instr = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["english"])
    audience_instr = AUDIENCE_INSTRUCTIONS.get(audience, "")
    custom_context = f"\nExtra context: {custom_prompt}" if custom_prompt else ""

    try:
        image_blocks = build_image_blocks(images)
        prompt = f"""Look at this image and write 2 very different Instagram captions for A/B testing.
{lang_instr}
Base style: {style["description"]}
{audience_instr}{custom_context}

Version A should be: SHORT and punchy (1 sentence max, under 80 chars), direct and bold
Version B should be: LONG and storytelling (3-4 sentences), emotional and detailed

Format exactly like this:
VERSION_A: [caption here]
VERSION_B: [caption here]

Rules:
- Include emojis naturally
- Do NOT include hashtags
- Make them feel completely different in tone and length
- Only output the two lines above, nothing else"""

        image_blocks.append({"type": "text", "text": prompt})
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": image_blocks}],
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        version_a = ""
        version_b = ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("VERSION_A:"):
                version_a = line[10:].strip()
            elif line.upper().startswith("VERSION_B:"):
                version_b = line[10:].strip()

        if not version_a:
            version_a = "✨ Moment captured."
        if not version_b:
            version_b = "Some moments are too beautiful to describe — they just need to be felt. This is one of those moments. ✨"

        return jsonify({
            "version_a": version_a,
            "version_b": version_b,
            "style": style["label"],
        })
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
