from flask import request, jsonify

from caption.services.generate_caption import generate_caption, regenerate_single_caption
from caption.services.generate_story import generate_story
from caption.services.generate_hashtags import generate_hashtags


def _collect_images():
    """Collect images from request — single or carousel."""
    images = request.files.getlist("images[]")
    single = request.files.get("image")
    if single:
        images = [single]
    return images


def generate_caption_view(client):
    """Handle POST /generate — captions, story, hashtags, regenerate."""
    try:
        style            = request.form.get("style", "casual")
        style2           = request.form.get("style2", "")
        audience         = request.form.get("audience", "general")
        language         = request.form.get("language", "english")
        length           = request.form.get("length", "medium")
        mood             = request.form.get("mood", "none")
        custom_prompt    = request.form.get("custom_prompt", "")
        hashtags_only    = request.form.get("hashtags_only", "false").lower() == "true"
        story_mode       = request.form.get("story_mode", "false").lower() == "true"
        regenerate_index = request.form.get("regenerate_index")

        images = _collect_images()
        if not images:
            return jsonify({"error": "No image provided"}), 400

        if hashtags_only:
            result = generate_hashtags(client, images, style, audience, custom_prompt)
            return jsonify(result)

        if story_mode:
            result = generate_story(client, images, style, style2, audience, language, mood, custom_prompt)
            return jsonify(result)

        if regenerate_index is not None:
            result = regenerate_single_caption(client, images, style, style2, audience, language, length, mood, custom_prompt)
            return jsonify(result)

        result = generate_caption(client, images, style, style2, audience, language, length, mood, custom_prompt)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
