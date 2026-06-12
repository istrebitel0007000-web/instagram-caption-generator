import os
import warnings
from flask import Flask
from groq import Groq
from dotenv import load_dotenv
import routes

load_dotenv()

app = Flask(__name__)

_api_key = os.getenv("GROQ_API_KEY", "")
if not _api_key:
    warnings.warn("GROQ_API_KEY is not set — API calls will fail at runtime.")
client = Groq(api_key=_api_key)

app.add_url_rule("/",         "index",        routes.index)
app.add_url_rule("/generate", "generate",     lambda: routes.generate(client),     methods=["POST"])
app.add_url_rule("/bio",      "generate_bio", lambda: routes.generate_bio(client), methods=["POST"])
app.add_url_rule("/analyze",  "analyze",      lambda: routes.analyze(client),      methods=["POST"])
app.add_url_rule("/ab_test",  "ab_test",      lambda: routes.ab_test(client),      methods=["POST"])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
