from flask import request, jsonify

from caption.services.generate_bio import generate_bio


def generate_bio_view(client):
    """Handle POST /bio — generate Instagram bios."""
    try:
        style         = request.form.get("style", "casual")
        audience      = request.form.get("audience", "general")
        language      = request.form.get("language", "english")
        custom_prompt = request.form.get("custom_prompt", "")

        result = generate_bio(client, style, audience, language, custom_prompt)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
