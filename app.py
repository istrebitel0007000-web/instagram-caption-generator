import os

from flask import Flask, render_template
from groq import Groq
from dotenv import load_dotenv

from caption.views.generate_caption import generate_caption_view
from caption.views.generate_bio import generate_bio_view
from caption.views.analyze_image import analyze_image_view
from caption.views.ab_test import ab_test_view

load_dotenv()

# ── API KEY VALIDATION ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    raise RuntimeError(
        "\n\n❌ GROQ_API_KEY is missing!\n"
        "  • Locally: create a .env file with GROQ_API_KEY=gsk_xxxx\n"
        "  • Render:   add it in Dashboard → Environment tab\n"
        "  Get your key at: https://console.groq.com/keys\n"
    )

client = Groq(api_key=GROQ_API_KEY)

# ── FLASK APP ───────────────────────────────────────────────────────
app = Flask(__name__)


# ── ROUTES ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    return generate_caption_view(client)


@app.route("/bio", methods=["POST"])
def bio():
    return generate_bio_view(client)


@app.route("/analyze", methods=["POST"])
def analyze():
    return analyze_image_view(client)


@app.route("/ab_test", methods=["POST"])
def ab_test():
    return ab_test_view(client)


if __name__ == "__main__":
    app.run(debug=True)
