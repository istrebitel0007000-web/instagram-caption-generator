from flask import request, jsonify

from caption.services.analyze_image import analyze_image


def analyze_image_view(client):
    """Handle POST /analyze — analyze image for mood, tags, description."""
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        result = analyze_image(client, image)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
