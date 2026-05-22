# 📸 Instagram Caption Generator

A smart AI-powered web app that generates Instagram captions, hashtags, bios, and Story content from uploaded photos — powered by Groq's vision model with multi-mode generation and 8-language support.

🌐 **Live:** [capgen.duckdns.org](http://capgen.duckdns.org)

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Framework | Flask 3.0.3 |
| Server | Gunicorn 22.0.0 |
| AI Provider | Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`) |
| Frontend | HTML + Jinja2 templates + JavaScript |
| Deployment | AWS EC2 (eu-north-1) + DuckDNS |
| Alt Deploy | Render.com (`render.yaml` included) |

---

## 📁 Project Structure

```
instagram-caption-generator/
├── app.py                  # All Flask routes and AI logic
├── templates/
│   └── index.html          # Main UI template
├── static/                 # CSS, JS, images
├── requirements.txt        # Python dependencies
├── render.yaml             # One-click Render deployment
├── test_oracle_brain.py    # Integration tests
├── test_r1_to_r9.py        # Route tests
└── .env                    # Environment variables (not committed)
```

---

## ⚡ API Endpoints

All endpoints accept `multipart/form-data` (for image uploads) or `application/x-www-form-urlencoded`.

### Pages

| Method | URL | Description |
|---|---|---|
| GET | `/` | Serves the main web UI |

### Generation

| Method | URL | Description |
|---|---|---|
| POST | `/generate` | Generate 3 captions, hashtags, story cards, or A/B test — controlled by form params |
| POST | `/bio` | Generate 3 Instagram bio options |
| POST | `/analyze` | Auto-detect image emotion, language, and suggest @tags |
| POST | `/ab_test` | Generate 2 versions (short punchy vs long storytelling) for A/B testing |

---

## 🎛️ Generation Modes

The `/generate` endpoint supports 4 distinct modes via form parameters:

| Mode | Trigger Param | Description |
|---|---|---|
| **Standard** | *(default)* | Returns 3 different caption variants |
| **Hashtags Only** | `hashtags_only=true` | Returns 20–25 relevant hashtags |
| **Story Mode** | `story_mode=true` | Returns 3 Story cards with caption + poll + question |
| **Regenerate One** | `regenerate_index=<0-2>` | Regenerates a single caption by index |

---

## 🎨 Styles

| Key | Label | Description |
|---|---|---|
| `casual` | Casual & Fun 😎 | Relaxed, fun, conversational with emojis |
| `aesthetic` | Aesthetic ✨ | Dreamy, poetic, visually descriptive |
| `motivational` | Motivational 💪 | Inspiring, energetic, uplifting |
| `funny` | Funny & Witty 😂 | Humorous, clever, playful with jokes/puns |
| `professional` | Professional 💼 | Polished, formal, business-appropriate |
| `romantic` | Romantic ❤️ | Loving, warm, heartfelt |

> Style mixing is supported — pass both `style` and `style2` to blend two styles.

---

## 📏 Caption Lengths

| Key | Description |
|---|---|
| `short` | 1–2 sentences, under 80 characters |
| `medium` | 2–3 sentences, ~100–180 characters |
| `long` | 3–5 sentences, ~200–350 characters |

---

## 😊 Moods

`none` · `happy` · `sad` · `mysterious` · `bold` · `chill` · `grateful`

---

## 👥 Target Audiences

`general` · `teens` · `professionals` · `fitness` · `foodies` · `travelers` · `parents` · `entrepreneurs` · `creatives`

---

## 🌍 Supported Languages

| Key | Language |
|---|---|
| `english` | English |
| `spanish` | Español |
| `french` | Français |
| `german` | Deutsch |
| `portuguese` | Português |
| `arabic` | العربية |
| `russian` | Русский |
| `uzbek` | O'zbek tili |

---

## 🤖 AI Model

| Model | Purpose |
|---|---|
| `meta-llama/llama-4-scout-17b-16e-instruct` | All generation — captions, hashtags, bio, analysis, stories, A/B |

The model receives the uploaded image(s) as base64-encoded `image_url` blocks alongside the prompt. Up to **4 images** are supported per request (carousel mode).

---

## 📋 Form Parameters Reference

### `/generate`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `images[]` | file[] | — | One or more images (max 4) |
| `image` | file | — | Single image (fallback if `images[]` absent) |
| `style` | string | `casual` | Primary caption style |
| `style2` | string | `null` | Optional secondary style to blend |
| `language` | string | `english` | Output language |
| `length` | string | `medium` | Caption length |
| `mood` | string | `none` | Emotional mood overlay |
| `audience` | string | `general` | Target audience |
| `custom_prompt` | string | `""` | Extra context for the AI |
| `hashtags_only` | bool | `false` | Return hashtags instead of captions |
| `story_mode` | bool | `false` | Return Story cards instead of captions |
| `regenerate_index` | int | `null` | Regenerate single caption at this index |

### `/bio`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `style` | string | `casual` | Bio style |
| `audience` | string | `general` | Target audience |
| `language` | string | `english` | Output language |
| `custom_prompt` | string | `""` | Extra details about the account |

### `/analyze`

| Parameter | Type | Description |
|---|---|---|
| `image` | file | Single image to analyze |

Returns: `language`, `emotion`, `emotion_confidence`, `tags[]`, `image_description`

### `/ab_test`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `images[]` / `image` | file | — | One or more images |
| `style` | string | `casual` | Base style |
| `language` | string | `english` | Output language |
| `audience` | string | `general` | Target audience |
| `custom_prompt` | string | `""` | Extra context |

Returns: `version_a` (short, ≤80 chars), `version_b` (long, 3–4 sentences)

---

## 🛠️ Local Setup

### 1. Clone and install

```bash
git clone https://github.com/istrebitel0007000-web/instagram-caption-generator.git
cd instagram-caption-generator
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.template .env
# Edit .env and add your Groq API key
```

`.env` contents:
```
GROQ_API_KEY=your_groq_api_key_here
PORT=5000
```

### 3. Run the development server

```bash
python app.py
```

App will be available at `http://localhost:5000`

---

## ☁️ Deploy to Render

This repo includes a `render.yaml` for one-click deployment.

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your `instagram-caption-generator` repo
4. Render auto-detects `render.yaml` and creates a web service running:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
5. Add your environment variable in the Render dashboard:
   - `GROQ_API_KEY` → your Groq API key

---

## 🖥️ Deploy to EC2 (Current Setup)

The app is currently running on **AWS EC2 t3.micro** in `eu-north-1` (Stockholm).

```bash
# On the instance
git clone https://github.com/istrebitel0007000-web/instagram-caption-generator.git
cd instagram-caption-generator
pip install -r requirements.txt

# Create .env with your GROQ_API_KEY

# Run with gunicorn (production)
gunicorn app:app --bind 0.0.0.0:5000 --daemon

# Or run with systemd service for auto-restart on reboot
```

**Instance details:**
- Instance ID: `i-05cce20cc765ea885`
- Type: `t3.micro`
- Region: `eu-north-1` (Stockholm)
- Public IP: `16.192.32.84` (Elastic IP)
- Domain: `capgen.duckdns.org` → DuckDNS pointing to the Elastic IP

---

## 🧪 Running Tests

```bash
python manage.py test test_oracle_brain
python manage.py test test_r1_to_r9
```

Or directly:

```bash
python -m pytest test_oracle_brain.py test_r1_to_r9.py -v
```

---

## 📦 Dependencies

```
flask==3.0.3
groq==0.28.0
python-dotenv==1.0.1
Werkzeug==3.0.3
gunicorn==22.0.0
```

---

## 🔐 Security Notes

- `GROQ_API_KEY` is loaded from `.env` — **never commit `.env` to git**
- Add `.env` and `__pycache__/` to your `.gitignore`
- The EC2 instance has **IMDSv2 required** (good practice)
- No IAM role is attached to the instance (least-privilege — correct if not using AWS services from the app)

---

## 📄 License

MIT
