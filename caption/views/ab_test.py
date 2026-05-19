from flask import request, jsonify

from caption.services.ab_test_caption import ab_test_caption


def ab_test_view(client):
    """Handle POST /ab_test — generate A/B test captions."""
    try:
        style         = request.form.get("style", "casual")
        style2        = request.form.get("style2", "")
        audience      = request.form.get("audience", "general")
        language      = request.form.get("language", "english")
        mood          = request.form.get("mood", "none")
        custom_prompt = request.form.get("custom_prompt", "")

        images = request.files.getlist("images[]")
        single = request.files.get("image")
        if single:
            images = [single]
        if not images:
            return jsonify({"error": "No image provided"}), 400

        result = ab_test_caption(client, images, style, style2, audience, language, mood, custom_prompt)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
