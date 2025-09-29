import os
import csv
import time
import json
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env beside this file

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError
from openai import AsyncOpenAI

# ========= ENV =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")  # safe default

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Missing TELEGRAM_TOKEN or OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ========= FILES & CONSTANTS =========
LOG_FILE = "usage_log.csv"
FACTS_FILE = "local_facts.json"
MAX_TG = 4000

# ========= DEFAULT SEED FACTS =========
FACTS_DEFAULT = {
    "dangkao": {
        "competitors": [
            "Peng Huoth Borey (Hun Sen Blvd corridor)",
            "Chip Mong Land near Cheung Aek",
            "Multiple small/medium private boreys in Prey Sa & Spean Thma"
        ],
        "prices": "Broad range ~$130–$450/m². Near AEON & main boulevards: ~$300–$350/m²; fringe/side roads: ~$150–$200/m².",
        "opportunity": "Mid-market link villas & 250–400 m² plots with good road/utilities; clarify financing & fast handover."
    },
    "cheung aek": {
        "competitors": [
            "Chip Mong housing projects",
            "Local private boreys along 271 extension & side streets"
        ],
        "prices": "Commonly ~$180–$360/m² depending on frontage and road width/access.",
        "opportunity": "‘Smart villa’ upgrade positioning (better finishing, sunlight/ventilation), add move-in packages."
    },
    "prey sa": {
        "competitors": [
            "Local land flippers",
            "Small boreys targeting factory staff and young families"
        ],
        "prices": "Land ~$140–$260/m² off main roads; closer to main corridors can be ~$240–$320/m².",
        "opportunity": "Affordable shophouses for micro-business + rent-to-own schemes; highlight commute times and schools."
    },
    "takhmao": {
        "competitors": [
            "Borey Peng Huoth (Takhmao)",
            "Family-built boreys & riverside boutique projects"
        ],
        "prices": "Residential land ~$250–$500/m² on main roads; inner areas ~$150–$250/m².",
        "opportunity": "Mid-market villas/shophouses for families relocating from central PP; emphasize livability & river proximity."
    },
    "mean chey": {
        "competitors": [
            "Chip Mong 271",
            "Peng Huoth along Hun Sen Blvd/60m",
            "Various mixed-use near AEON Mean Chey"
        ],
        "prices": "Prime corridors ~$500–$1,200/m²; inner streets ~$250–$400/m².",
        "opportunity": "Mixed-use (home+business), inventory for upgrade buyers; promote mall/school/road access advantages."
    },
    "chbar ampov": {
        "competitors": [
            "Borey Peng Huoth The Star & premium estates",
            "High-end gated communities"
        ],
        "prices": "Main boulevards ~$700–$1,500/m²; inner plots ~$400–$600/m².",
        "opportunity": "Upper segment & riverside lifestyle; position for status, privacy, and quality of life."
    }
}

# ========= FACTS STORAGE =========
def save_facts(facts: dict) -> None:
    with open(FACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)

def load_facts() -> dict:
    if not os.path.exists(FACTS_FILE):
        save_facts(FACTS_DEFAULT)
        return dict(FACTS_DEFAULT)
    try:
        with open(FACTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("facts is not a dict")
            return data
    except Exception:
        save_facts(FACTS_DEFAULT)
        return dict(FACTS_DEFAULT)

FACTS = load_facts()

# ========= HELPERS =========
def log_usage(user_id, username, message):
    try:
        new_file = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp_iso", "user_id", "username", "message"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), user_id, username or "", message])
    except Exception as e:
        print(f"⚠️ Logging error: {e}")

def detect_language(text: str) -> str:
    if text:
        for ch in text:
            if "\u1780" <= ch <= "\u17FF":
                return "khmer"
    return "english"

def system_prompt_for(lang: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    if lang == "khmer":
        return (
            "អ្នកគឺជាគ្រូបណ្តុះបណ្តាលលក់អចលនទ្រព្យជាន់ខ្ពស់សម្រាប់ទីផ្សារកម្ពុជា។ "
            "ឆ្លើយតែជាភាសាខ្មែរ បែបជំហានៗ។ "
            f"ថ្ងៃនេះ៖ {today}."
        )
    else:
        return (
            "You are a senior real-estate sales trainer for Cambodia. "
            "Reply ONLY in English, step-by-step, practical. "
            f"Today's date: {today}."
        )

def normalize_name(raw: str) -> str:
    return (raw or "").strip().lower()

def normalized_collapse(s: str) -> str:
    """lowercase and remove non-alphanumeric to improve matching"""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())

def find_fact_key_by_text(text: str) -> Optional[str]:
    """Find a saved FACTS key mentioned in free text (supports spacing variations)."""
    t = (text or "").lower()
    for k in FACTS.keys():
        if k in t:
            return k
    t2 = normalized_collapse(text or "")
    for k in FACTS.keys():
        if normalized_collapse(k) in t2:
            return k
    return None

def intent_from_text(text: str) -> Optional[str]:
    """
    Map free-text to a field:
      - 'competitors' if user asks about rivals
      - 'prices' if asking price/promo
      - 'opportunity' if asking opportunity/USP
      - 'insight' if asking generally
    """
    t = (text or "").lower()
    comp_words = ["competitor", "competitors", "vs", "rival", "alternative", "who else"]
    price_words = ["price", "prices", "$", "usd", "៛", "promo", "promotion", "discount", "start", "starts", "from"]
    opp_words = ["opportunity", "usp", "why buy", "strength", "gap", "positioning", "target"]
    insight_words = ["insight", "overview", "around", "near", "market", "detail", "details", "info"]

    if any(w in t for w in comp_words): return "competitors"
    if any(w in t for w in price_words): return "prices"
    if any(w in t for w in opp_words): return "opportunity"
    if any(w in t for w in insight_words): return "insight"
    return None

def choose_field_by_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["price", "prices", "starts", "start", "from", "$", "usd", "៛", "promo", "promotion", "discount"]):
        return "prices"
    if any(k in t for k in ["competitor", "competitors", "near", "around", "developer", "borey", "chip mong", "peng huoth"]):
        return "competitors"
    return "opportunity"

def merge_competitors(existing: list, new_text: str) -> list:
    # Split by ; or , and merge unique
    parts = []
    for seg in new_text.replace("；", ";").replace("，", ",").split(";"):
        parts.extend([p.strip() for p in seg.split(",")])
    parts = [p for p in parts if p]
    base = set([c.strip() for c in (existing or []) if c and c.strip()])
    for c in parts:
        if c not in base:
            existing.append(c)
            base.add(c)
    return existing

async def ai_chat(user_text: str, temperature: float = 0.25) -> str:
    lang = detect_language(user_text)
    try:
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_for(lang)},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return ("⚠️ មិនអាចភ្ជាប់ទៅម៉ាស៊ីនមេបានទេ" if lang == "khmer" else "⚠️ Couldn’t reach the AI") + f" Error: {e}"

async def send_long(update: Update, text: str):
    if not text:
        return
    for i in range(0, len(text), MAX_TG):
        await update.message.reply_text(text[i:i + MAX_TG])

# ========= INSIGHT & FACTS COMMANDS =========
async def insight(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /insight <name or area>
    if context.args:
        key = normalize_name(" ".join(context.args))
    else:
        key = normalize_name((update.message.text or "").replace("/insight", ""))

    if not key:
        await update.message.reply_text("⚠️ Usage: /insight <area or project>. Example: /insight Dangkao")
        return

    match = None
    if key in FACTS:
        match = key
    else:
        for k in FACTS.keys():
            if k in key:
                match = k
                break

    if not match:
        await update.message.reply_text(
            f"❌ No facts yet for: {key}.\nAdd it via /addfact {key} | competitors=... | prices=... | opportunity=..."
        )
        return

    data = FACTS[match]
    competitors = ", ".join(data.get("competitors", [])) if data.get("competitors") else "—"
    prices = data.get("prices", "—")
    opp = data.get("opportunity", "—")

    text = (
        f"📊 **Local Insight: {match.title()}**\n\n"
        f"🏢 Competitors: {competitors}\n\n"
        f"💲 Price Range: {prices}\n\n"
        f"🚀 Opportunity: {opp}"
    )
    await send_long(update, text)

def parse_kv_parts(raw: str) -> dict:
    """
    Parse 'name | competitors=... | prices=... | opportunity=...'
    Returns dict with keys: name, competitors(list), prices(str), opportunity(str)
    """
    out = {"name": "", "competitors": None, "prices": None, "opportunity": None}
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if parts:
        out["name"] = parts[0]
    for part in parts[1:]:
        if "=" in part:
            key, val = part.split("=", 1)
        elif ":" in part:
            key, val = part.split(":", 1)
        else:
            continue
        key = key.strip().lower()
        val = val.strip()
        if key == "competitors":
            items = [x.strip() for x in val.replace("၊", ",").replace("；", ";").replace("，", ",").split(";")]
            if len(items) == 1:
                items = [x.strip() for x in val.split(",")]
            out["competitors"] = [x for x in items if x]
        elif key in ("prices", "price", "pricing"):
            out["prices"] = val
        elif key in ("opportunity", "opp"):
            out["opportunity"] = val
    return out

async def addfact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /addfact <name> | competitors=... | prices=... | opportunity=...
    """
    raw = (update.message.text or "")
    after = raw.split(" ", 1)
    if len(after) < 2 or not after[1].strip():
        await update.message.reply_text(
            "⚠️ Usage:\n/addfact <name> | competitors=... | prices=... | opportunity=...\n"
            "Example:\n/addfact Lucky One | competitors=Nearby boreys; Peng Huoth 271 | prices=~$95k promo | opportunity=First-home buyers"
        )
        return

    fields = parse_kv_parts(after[1])
    name = normalize_name(fields.get("name", ""))
    if not name:
        await update.message.reply_text("⚠️ Please include a name: /addfact <name> | ...")
        return

    entry = FACTS.get(name, {})
    if fields.get("competitors") is not None:
        entry["competitors"] = fields["competitors"]
    if fields.get("prices") is not None:
        entry["prices"] = fields["prices"]
    if fields.get("opportunity") is not None:
        entry["opportunity"] = fields["opportunity"]
    FACTS[name] = entry
    save_facts(FACTS)

    comp_str = ", ".join(entry.get("competitors", [])) if entry.get("competitors") else "—"
    prices = entry.get("prices", "—")
    opp = entry.get("opportunity", "—")
    await update.message.reply_text(
        f"✅ Saved **{name.title()}**\n🏢 Competitors: {comp_str}\n💲 Prices: {prices}\n🚀 Opportunity: {opp}"
    )

async def remember(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /remember <free text>
    Natural language memory. Heuristics:
      - The first 1–4 words before a price/opportunity keyword are treated as the project/area name.
      - Classify the rest as prices / competitors / opportunity.
      - Append (don’t overwrite) into local_facts.json.
    Also supports structured: /remember Name | prices=... (or competitors=..., opportunity=...)
    """
    raw = (update.message.text or "")
    payload = raw.split(" ", 1)
    if len(payload) < 2 or not payload[1].strip():
        await update.message.reply_text(
            "⚠️ Usage:\n/remember Lucky One starts from $90,000 with promotion free hard title\n"
            "or structured:\n/remember Lucky One | prices=from $90,000 with free hard title"
        )
        return

    text = payload[1].strip()

    # Structured path
    if "|" in text and ("=" in text or ":" in text):
        fields = parse_kv_parts(text)
        name = normalize_name(fields.get("name", ""))
        if not name:
            await update.message.reply_text("⚠️ Please include a name before the first '|'.")
            return
        entry = FACTS.get(name, {})
        if fields.get("competitors"):
            entry["competitors"] = merge_competitors(entry.get("competitors", []), ", ".join(fields["competitors"]))
        if fields.get("prices"):
            entry["prices"] = (entry.get("prices", "") + " | " if entry.get("prices") else "") + fields["prices"]
        if fields.get("opportunity"):
            entry["opportunity"] = (entry.get("opportunity", "") + " | " if entry.get("opportunity") else "") + fields["opportunity"]
        FACTS[name] = entry
        save_facts(FACTS)
        await update.message.reply_text(f"✅ Remembered for **{name.title()}**.")
        return

    # Heuristic parse for: /remember Lucky One starts from $90,000 ...
    boundary_tokens = {"starts", "start", "from", "price", "prices", "promotion", "promo", "opportunity", "near", "around"}
    tokens = text.split()
    name_words, rest_start_idx = [], 0
    for i, tok in enumerate(tokens):
        t = tok.lower()
        if t in boundary_tokens or "$" in t or "usd" in t or "៛" in t:
            rest_start_idx = i
            break
        if len(name_words) < 4:
            name_words.append(tok)
            rest_start_idx = i + 1
        else:
            break

    name = normalize_name(" ".join(name_words))
    rest = " ".join(tokens[rest_start_idx:]).strip()

    if not name:
        await update.message.reply_text("⚠️ I couldn't detect the project/area name. Try: /remember <Name> starts from $90,000 ...")
        return
    if not rest:
        await update.message.reply_text("⚠️ Please include some details after the name.")
        return

    field = choose_field_by_text(rest)
    entry = FACTS.get(name, {})

    if field == "competitors":
        entry["competitors"] = merge_competitors(entry.get("competitors", []), rest)
    elif field == "prices":
        entry["prices"] = (entry.get("prices", "") + " | " if entry.get("prices") else "") + rest
    else:
        entry["opportunity"] = (entry.get("opportunity", "") + " | " if entry.get("opportunity") else "") + rest

    FACTS[name] = entry
    save_facts(FACTS)

    await update.message.reply_text(
        f"✅ Remembered for **{name.title()}**\n"
        f"🗂 Field: {field}\n"
        f"📝 Saved: {rest}"
    )

async def showfact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /showfact <name>
    name = normalize_name(" ".join(context.args)) if context.args else ""
    if not name:
        await update.message.reply_text("⚠️ Usage: /showfact <name>")
        return
    key = None
    if name in FACTS:
        key = name
    else:
        for k in FACTS.keys():
            if k in name or name in k:
                key = k
                break
    if not key:
        await update.message.reply_text(f"❌ Not found: {name}")
        return
    data = FACTS[key]
    comp = ", ".join(data.get("competitors", [])) if data.get("competitors") else "—"
    prices = data.get("prices", "—")
    opp = data.get("opportunity", "—")
    await update.message.reply_text(f"📘 **{key.title()}**\n🏢 {comp}\n💲 {prices}\n🚀 {opp}")

async def listfacts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not FACTS:
        await update.message.reply_text("No facts saved yet.")
        return
    keys = sorted([k.title() for k in FACTS.keys()])
    await send_long(update, "📚 Saved areas/projects:\n- " + "\n- ".join(keys))

async def delfact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /delfact <name>
    name = normalize_name(" ".join(context.args)) if context.args else ""
    if not name:
        await update.message.reply_text("⚠️ Usage: /delfact <name>")
        return
    key = None
    if name in FACTS:
        key = name
    else:
        for k in FACTS.keys():
            if k in name or name in k:
                key = k
                break
    if not key:
        await update.message.reply_text(f"❌ Not found: {name}")
        return
    del FACTS[key]
    save_facts(FACTS)
    await update.message.reply_text(f"🗑️ Deleted: {key.title()}")

# ========= BASIC & VOICE =========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Sales Coach Bot ready.\n"
        "Commands:\n"
        "• /ask <question>  (or just type)\n"
        "• /dailytip\n"
        "• /dailyjob\n"
        "• /insight <area or project>\n"
        "• /addfact <name> | competitors=... | prices=... | opportunity=...\n"
        "• /remember <free text about a project> (auto-save)\n"
        "• /showfact <name>\n"
        "• /listfacts\n"
        "• /delfact <name>\n"
        "• /report\n"
        "Send a voice note for auto transcription + answer."
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = " ".join(context.args).strip() if context.args else (update.message.text or "").strip()
    if not q:
        await update.message.reply_text("Please type your question.")
        return

    # Log every query
    log_usage(update.effective_user.id, update.effective_user.username, q)

    # Smart-route plain text to facts if possible
    key = find_fact_key_by_text(q)
    if key:
        intent = intent_from_text(q)
        data = FACTS.get(key, {})
        comp = ", ".join(data.get("competitors", [])) if data.get("competitors") else "—"
        prices = data.get("prices", "—")
        opp = data.get("opportunity", "—")

        if intent == "competitors":
            await send_long(update, f"🏢 Competitors for {key.title()}: {comp}")
            return
        elif intent == "prices":
            await send_long(update, f"💲 Prices for {key.title()}: {prices}")
            return
        elif intent == "opportunity":
            await send_long(update, f"🚀 Opportunity for {key.title()}: {opp}")
            return
        elif intent == "insight":
            text = (
                f"📊 Local Insight: {key.title()}\n\n"
                f"🏢 Competitors: {comp}\n\n"
                f"💲 Price Range: {prices}\n\n"
                f"🚀 Opportunity: {opp}"
            )
            await send_long(update, text)
            return
        else:
            # No clear intent: let AI answer, but ground with saved facts
            facts_block = (
                f"Facts for {key.title()}:\n"
                f"- Competitors: {comp}\n"
                f"- Prices: {prices}\n"
                f"- Opportunity: {opp}\n"
            )
            answer = await ai_chat(q + "\n\nUse these facts:\n" + facts_block)
            await send_long(update, answer)
            return

    # Fallback to coaching AI
    answer = await ai_chat(q)
    await send_long(update, answer)

async def dailytip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = detect_language(update.message.text or "")
    prompt = "ផ្តល់ពាក្យលើកទឹកចិត្ត" if lang == "khmer" else "Give one motivational sales quote"
    answer = await ai_chat(prompt, temperature=0.2)
    await send_long(update, "🌟 Daily Sales Quote\n\n" + answer)

async def dailyjob(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = detect_language(update.message.text or "")
    prompt = "រៀបចំកិច្ចការប្រចាំថ្ងៃ" if lang == "khmer" else "Design one focused daily plan"
    answer = await ai_chat(prompt, temperature=0.2)
    await send_long(update, "📌 Daily Sales Plan\n\n" + answer)

async def report_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("No log yet.")
        return
    with open(LOG_FILE, "rb") as f:
        await update.message.reply_document(document=f, filename=LOG_FILE)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    if not voice:
        return

    tg_file = await context.bot.get_file(voice.file_id)
    local_path = "voice.ogg"

    try:
        await tg_file.download_to_drive(local_path)

        async def transcribe_with(model_name: str):
            with open(local_path, "rb") as f:
                return await client.audio.transcriptions.create(model=model_name, file=f)

        try:
            tr = await transcribe_with(TRANSCRIBE_MODEL)
        except Exception:
            tr = await transcribe_with("whisper-1")  # fallback

        text = (getattr(tr, "text", None) or "").strip()
        if not text:
            await update.message.reply_text("⚠️ Couldn’t understand the audio.")
            return

        log_usage(update.effective_user.id, update.effective_user.username, f"[VOICE] {text}")
        answer = await ai_chat(text)
        await send_long(update, answer)

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error transcribing audio: {e}")
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass

# ========= MAIN =========
def main():
    request = HTTPXRequest(connect_timeout=30, read_timeout=90)
    app = Application.builder().token(TELEGRAM_TOKEN).request(request).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("dailytip", dailytip))
    app.add_handler(CommandHandler("dailyjob", dailyjob))
    app.add_handler(CommandHandler("insight", insight))
    app.add_handler(CommandHandler("addfact", addfact))
    app.add_handler(CommandHandler("remember", remember))  # NEW
    app.add_handler(CommandHandler("showfact", showfact))
    app.add_handler(CommandHandler("listfacts", listfacts))
    app.add_handler(CommandHandler("delfact", delfact))
    app.add_handler(CommandHandler("report", report_csv))

    # Free text as /ask
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask))
    # Voice notes
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    while True:
        try:
            app.run_polling(drop_pending_updates=True)
            break
        except (TimedOut, NetworkError) as e:
            print(f"⚠️ Network issue: {e}. Retrying…")
            time.sleep(5)

if __name__ == "__main__":
    main()
