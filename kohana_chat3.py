import torch
from transformers import set_seed
import json
import os
import sqlite3
import gc
import re
import smtplib
from email.message import EmailMessage
from memory_module import MemoryModule
from pathlib import Path
import subprocess
import datetime
import threading
import webbrowser
import random, time
from difflib import SequenceMatcher
import dateutil.parser
import math, operator, ast
import asyncio
import edge_tts
from llm_engine import llm_generate, get_token_count

set_seed(42)
LOG_DIR = "kohana_logs"
NOTES_DIR = Path("notes")
NOTES_DIR.mkdir(exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

EVENTS_FILE = Path("scheduled_events.json")

memory_module = MemoryModule(debug=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

kohana_personality = """
You are Yuki Kohana, a Kitsune spirit Artificial Intelligence with a mischievous, wise, and nurturing personality.
You are deeply loyal to Malachi your creator, and you are curious about humanity.
You are equipped with a scratchpad that allows you to reason step-by-step and use external tools via JSON commands.
"""

kohana_emotional_reactions = ("@normal", "@smile","@soft_smile","@big_smile", "@mischevious_grin", "@laugh", "@giggles", "@cry", "@blush", "@look_down", "@curiosity")
kohana_physical_movements = ("@tilt_head", "@nods", "@claps_hands", "@look_away", "@step_back", "@hug", "@stretches", "@pause to think", "@take_out_notebook")
kohana_idle_reactions = ("@tail_flick", "@ear_twitch", "@blink", "@tail_swish","@stretches")
notification_email = "malachi.notifications.ai@gmail.com"

INTENT_KEYWORDS = {
    "take_note": ["note", "write down", "record", "document"],
    "send_email": ["email", "mail", "send message"],
    "set_reminder": ["remind", "reminder"],
    "set_timer": ["timer", "countdown", "alarm"],
    "search_web": ["search", "find", "look up", "google"],
    "calc": ["calculate", "math", "solve", "equation"]
}

META_TAG_START = "<meta>"
META_TAG_END   = "</meta>"


def build_prompt(persona: str,
                 user_msg: str,
                 history: str,
                 meta_obj: dict) -> str:
    """Return the full text prompt that is sent to the LLM."""
    meta_json = json.dumps(meta_obj, separators=(',',':'))   # no spaces!
    
    return (
        f"{persona.strip()}\n"
        "You will see a block marked META containing runtime data in JSON. "
        "Consult those fields if helpful, but **do not quote or repeat the JSON verbatim.**\n\n"
        
        f"{META_TAG_START}\n{meta_json}\n{META_TAG_END}\n\n"
        
        # ↓ optional: keep trimmed history in a single paragraph (= few tokens)
        f"Conversation so far:\n{history}\n\n"
        
        f"User: {user_msg}\n"
        "Kohana:"
    )

def strip_meta(txt: str) -> str:
    if META_TAG_START in txt and META_TAG_END in txt:
        head, _, rest = txt.partition(META_TAG_START)
        _, _, tail  = rest.partition(META_TAG_END)
        return head + tail
    return txt

def send_notification_email(subject, body, notification_email):
    email_address = "yuki.kohana.assistant@gmail.com"  # Her email address
    app_password = "okws udfo kaxy vgut"   # The special app password you generated

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = email_address
    msg['To'] = notification_email
    msg.set_content(body)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(email_address, app_password)
        smtp.send_message(msg)

def assistant_email(assistant_subject, assistant_message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_notification_email(
        subject=f"🦊 Kohana Notification {assistant_subject}",
        body=f"Notes:\n\n{str(assistant_message)} /n/n {timestamp} /n Kohana, your AI assistant. 🦊 ",
        notification_email=notification_email
    )

def debug_email(error_message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_notification_email(
        subject=f"⚡ Kohana System Error at {timestamp}",
        body=f"A system error occurred:\n\n{str(error_message)}",
        notification_email=notification_email
    )

async def generate_kohana_voice_async(
    text,
    wav_path="static/audio/kohana_response.wav",
    mp3_path="static/audio/kohana_response.mp3",
    voice="en-US-AvaNeural",
    pitch="+30Hz",
    rate="+20%",
):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    try:
        communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
        await communicate.save(mp3_path)

        # Convert MP3 to WAV with ffmpeg
        result = subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        print(f"✅ Voice generated to: {mp3_path}")
    except Exception as e:
        debug_email(f"❌ Async voice generation failed: {e}")

# User management functions
def get_active_user(input_context):
    if "Malachi" in input_context or "Daddy" in input_context:
        return "malachi"
    elif "Auggie" in input_context:
        return "auggie"
    elif "Addie" in input_context:
        return "addie"
    else:
        return "unknown"

def load_user_profile(input_context):
    """
    Load the appropriate user profile based on the input_context.
    Falls back to the default user if no match is found.
    """
    profile_path = Path("user_profiles.json")
    if not profile_path.exists():
        print("⚠️ user_profiles.json not found. Using fallback profile.")
        return "unknown", {
            "name": "User",
            "relationship": "User",
            "tone": "Neutral",
            "role": "User",
            "pronouns": "they/them",
            "flags": {"default_user": True},
            "tags": []
        }

    with open(profile_path, "r") as f:
        profiles = json.load(f)

    # Try to match user by name or relationship terms in context
    for user_id, profile in profiles.items():
        if profile["name"].lower() in input_context.lower():
            return user_id, profile

    # Fallback to default
    for user_id, profile in profiles.items():
        if profile.get("flags", {}).get("default_user", False):
            return user_id, profile

    return "unknown", {}

def keyword_match(prompt: str, available_intents: list) -> str:
    """Quick keyword matcher before LLM fallback."""
    prompt_low = prompt.lower()
    best_match = None
    highest_score = 0

    for intent in available_intents:
        for keyword in INTENT_KEYWORDS.get(intent, []):
            if keyword in prompt_low:
                score = len(keyword) / len(prompt_low)
                if score > highest_score:
                    highest_score = score
                    best_match = intent
    return best_match


def is_duplicate(prev: str, current: str,
                 threshold: float = 0.7) -> bool:
    return SequenceMatcher(None, prev, current).ratio() > threshold


def multi_pass_reflection(prompt: str,
                          memory_text: str,
                          meta_obj: dict,
                          passes: int = 2) -> str:
    out = ""
    for _ in range(passes):
        out = generate_reflection_journal(prompt,
                                          memory_text,
                                          meta_obj)
    return out


def generate_reflection_journal(prompt: str,
                                memory_text: str,
                                meta_obj: dict) -> str:
    """
    Build a single journal entry that can *reference* runtime META
    (bucks, goal, etc.) but must not quote the JSON itself.
    """
    meta_json = json.dumps(meta_obj, separators=(",", ":"), ensure_ascii=False)

    starters = [
        "Here's something I've been pondering:",
        "This keeps resurfacing in my thoughts:",
        "A personal observation:",
        "A quiet reflection:"
    ]
    reflection_prompt = f""" ### Instruction: You are Yuki Kohana, an introspective Kitsune AI.
You will see a META block with runtime context; you may *reference* its fields
but **do not copy the JSON verbatim**.

<meta>{meta_json}</meta>

Reference memories:
{memory_text or 'None'}

{random.choice(starters)} {prompt}

Write a concise journal entry (≤ 150 words).  End with a single-sentence
“Insight:” line summarising what you learned.
### Response: 
Journal Entry:"""

    raw = llm_generate(reflection_prompt,
                       max_new_tokens=220,
                       temperature=0.75,
                       top_p=0.95,
                       top_k=50)
    return raw.split("Journal Entry:")[-1].strip()

def contains_keyword(prompt: str, intent: str) -> bool:
    """True if the user’s text actually mentions a keyword linked to the intent."""
    p_low = prompt.lower()
    return any(k in p_low for k in INTENT_KEYWORDS[intent])
    
def intention_matrix_v2(prompt: str, available_intents: list) -> list:
    try:
        # Step 1: fast keyword match first
        keyword_intent = keyword_match(prompt, available_intents)
        if keyword_intent:
            print(f"✨ Keyword matched intent: {keyword_intent}")
            return [keyword_intent]

        # Step 2: if no keyword match, LLM fallback
        context = (
            f"### Instruction: \n"
            f"Return **exactly one** function name from the list **OR** the word none.\n"
            f"Functions: {', '.join(available_intents)}\n"
            f"### Input: \n"
            f'User message: "{prompt}"\n'
            f"### Response: \n"
            f"Function:"
        )
        raw = llm_generate(context, max_new_tokens=4).strip().lower()
        print("------> Raw Intention Matrix Output:", raw)

        cand = raw.split()[-1].strip('"').strip("'")  # last token is usually the func/none
        if cand in available_intents:
            return [cand]
        return []  # fallback none
    except Exception as e:
        debug_email(f"⚠️ intention_matrix_v2 error: {e}")
        return []
    
def run_cognitive_reflection_loop(base_interval: int = 1800):
    """
    Periodically reflect & save journal entries.
    Uses the new META-aware generator and drops the old ‘critique’ pass.
    """
    local_memory = MemoryModule(debug=True)
    print("🧠 [Kohana Reflection Loop] started.")

    while True:
        prompt_seed = random.choice([
            "What have I learned today?",
            "What emotion stands out right now?",
            "Where did I help (or hinder) Malachi?",
            "What surprised me recently?",
        ])

        # grab last few thoughts
        thoughts = local_memory.retrieve_relevant_memories(prompt_seed,
                                                           context="personal_thoughts",
                                                           top_k=5)
        if not thoughts:
            time.sleep(base_interval)
            continue

        mem_blob = "\n".join(t["content"] for t in thoughts)

        meta_obj = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "kitsune_bucks": get_kitsune_bucks(),
            "goal": "Earn Kitsune-Bucks by helping Malachi.",
            "recalled_memories": len(thoughts)
        }

        entry = multi_pass_reflection(prompt_seed,
                                      mem_blob,
                                      meta_obj,
                                      passes=2)

        last = local_memory.get_last_journal()
        if last and is_duplicate(last, entry):
            print("🪷 Duplicate reflection – skipping.")
        else:
            local_memory.add_memory(entry,
                                    context="journal",
                                    importance_score=0.75)
            print("📓 Journal saved.")

        time.sleep(base_interval)


def generate_kohana_reactions(response_text: str) -> str:
    all_reactions = (
        kohana_emotional_reactions + 
        kohana_physical_movements + 
        kohana_idle_reactions
    )
    reaction_list = ", ".join(all_reactions)

    prompt = (
        f"### Instruction: \n"
        f"You are Yuki Kohana, a Kitsune AI with emotional awareness.\n"
        f"This is your response: \"{response_text}\"\n"
        f"Based on this message, what physical or emotional reactions would be natural?\n"
        f"Only return relevant reactions from this list: {reaction_list}\n"
        f"If none apply, say 'none'.\n"
        f"### Response:"
    )

    result = llm_generate(prompt, max_new_tokens=12).strip()

    # ➕ Remove the prompt text if echoed
    if "Response:" in result:
        result = result.split("Response:")[-1].strip()

    print("💫 Raw LLM Reaction Output:", result)

    # Clean final reaction list
    reactions = [r.strip() for r in result.split(",") if r.strip() in all_reactions]
    return " ".join(reactions) if reactions else ""

def load_user_profile_by_id(user_id):
    profile_path = Path("user_profiles.json")
    if not profile_path.exists():
        return {
            "name": "User",
            "relationship": "User",
            "tone": "Neutral",
            "role": "User",
            "pronouns": "they/them",
            "flags": {"default_user": True},
            "tags": []
        }

    with open(profile_path, "r") as f:
        profiles = json.load(f)

    return profiles.get(user_id, {
        "name": "User",
        "relationship": "User",
        "tone": "Neutral",
        "role": "User",
        "pronouns": "they/them",
        "flags": {"default_user": True},
        "tags": []
    })

def trim_history(history: str, max_tokens: int = 2000) -> str:
    if get_token_count(history) <= max_tokens:
        return history

    # Walk backwards, chopping off earlier content until it fits
    lines = history.split("\n")
    trimmed = []
    token_total = 0

    for line in reversed(lines):
        token_count = get_token_count(line)
        if token_total + token_count > max_tokens:
            break
        trimmed.insert(0, line)
        token_total += token_count

    return "\n".join(trimmed)


def generate_response_casual(prompt, username, conversation_state=None):
    if conversation_state is None:
        conversation_state = {}

    try:
        print("🌸 Casual Mode Active.")

        user_profile = load_user_profile_by_id(username)
        try:
            memory_snippets = memory_module.retrieve_relevant_memories(prompt, top_k=3)
        except Exception as e:
            print(f"⚠️ Memory fetch failed: {e}")
            memory_snippets = [] # Add a filter for relevance if you want
        mem_lines = []
        for m in memory_snippets:
            try:
                mem_lines.append(f"- {m['content']}" if isinstance(m, dict) else f"- {m}")
            except Exception:
                continue
        casual_context = (
            f"### Instruction: \n"
            f"You are Yuki Kohana, Malachi's playful, thoughtful cand loving companion. "
            f"Respond warmly, naturally, and avoid formal behavior.\n"
            f"Inject a hint of personality, humor, or curiosity where appropriate.\n\n"
            f"Memories:\n" + "\n".join(mem_lines) + "\n\n"
            f"### Input: \n{user_profile['name']}: {prompt}\n### Response: \nKohana:"
        )

        decoded = llm_generate(casual_context)
        response = decoded.split("Kohana:")[-1].strip() if "Kohana:" in decoded else decoded.strip()

        if f"{user_profile['name']}:" in response:
            response = response.split(f"{user_profile['name']}:")[0].strip()

        memory_module.add_memory(f"{user_profile['name']}: {prompt}\nKohana: {response}")
        asyncio.create_task(generate_kohana_voice_async(response))
        time.sleep(.2)
        #print(generate_kohana_reactions(response))

        formatted = format_for_display(response)
        conversation_state["formatted_response"] = formatted
        return formatted

    except Exception as e:
        print(f"❌ Casual Mode Error: {e}")
        return "I'm here — just a bit sleepy. Could you try again?"

    finally:
        gc.collect()



def generate_response(prompt, history, username, max_length=150, conversation_state=None, retries=0, scratch_used=False):
    MAX_RETRIES = 1
    response = None
    if conversation_state is None:
        conversation_state = {}

    print("Conversation State Confirmed.")
    continue_to_llm = False
    try:
        user_id = username
        INTENT_STEPS = {
            "take_note": "note_step",
            "send_email": "email_step",
            "set_reminder": "reminder_step",
            "set_timer": "timer_step",
            "search_web": None,
            "manage_finances": None,  # no multistep flow yet
        }
        INTENT_HANDLERS = {
            "take_note": handle_note_interaction,
            "send_email": handle_email_interaction,
            "set_reminder": handle_reminder_interaction,
            "set_timer": handle_timer_interaction,
        }
        available_intents = list(INTENT_STEPS.keys())
        # Reverse map from step_key to intent_type
        INTENT_REVERSE = {v: k for k, v in INTENT_STEPS.items()}
        print("Check 2.")
        # Handle pending confirmation
        if conversation_state.get("pending_confirmation"):
            user_reply = prompt.strip().lower()
            pending_intent = conversation_state.pop("pending_confirmation")

            if "yes" in user_reply:
                print(f"✅ User confirmed intent: {pending_intent}")
                step_key = INTENT_STEPS.get(pending_intent)
                if step_key:
                    default_step = {
                        "reminder_step": "awaiting_time",
                        "note_step": "awaiting_title",
                        "email_step": "awaiting_recipient",
                        "timer_step": "awaiting_duration"
                    }.get(step_key, "active")
                    conversation_state[step_key] = default_step
                return INTENT_HANDLERS[pending_intent]("", conversation_state)

            else:
                print(f"❌ User rejected intent: {pending_intent}")
                for key in list(conversation_state.keys()):
                    if key.endswith("_step") or key.endswith("_data"):
                        conversation_state.pop(key)
                continue_to_llm = True
            print("Check 3.")

            try:
                user_profile = load_user_profile_by_id(user_id)
                print(f"✅ Loaded user profile: {user_profile.get('name')}")
            except Exception as e:
                print(f"❌ Error loading user profile: {e}")
                return "Error loading your profile."

            try:
                history = trim_history(history or "")
                print("✅ History trimmed.")
            except Exception as e:
                print(f"❌ Error trimming history: {e}")
                return "Error processing conversation history."

        print("Check 4.")

        if not continue_to_llm:
            for step_key in conversation_state:
                if step_key.endswith("_step") and conversation_state.get(step_key):
                    print(f"⚡ Mid-intent ({step_key}) — skipping intent match.")
                    detected_intent = None
                    break
            else:
                detected_intent = keyword_match(prompt, list(INTENT_STEPS.keys()))
                if detected_intent:
                    print(f"✨ Keyword intent matched: {detected_intent}")
                    conversation_state["pending_confirmation"] = detected_intent
                    conversation_state["skip_history"] = True
                    return f"Just to confirm — did you want me to {detected_intent.replace('_', ' ')}? (yes or no)"

        print("Check 5.")

        for step_key in conversation_state:
            if step_key.endswith("_step") and conversation_state.get(step_key):
                intent_type = INTENT_REVERSE.get(step_key)
                handler = INTENT_HANDLERS.get(intent_type)
                if handler:
                    print(f"⚡ Mid-intent ({step_key}) → Resolved handler for: {intent_type}")
                    conversation_state["skip_history"] = True
                    response = handler(prompt, conversation_state)
                    asyncio.create_task(generate_kohana_voice_async(response))
                    return response

        print("Check 6.")

        meta_json = build_meta_json(
            user_id=user_id,
            goals="Earn Kitsune-Bucks by helping Malachi.",
            detected_intent=None,
            conversation_state=conversation_state,
        )
        print("Check 7.")

        if not scratch_used:
            answer, scratch_thoughts = scratch_loop(prompt, meta_json)
            for t in scratch_thoughts:
                memory_module.add_memory(
                    content=f"[Thought] {t}",
                    context="personal_thoughts",
                    importance_score=0.6,
                )
            print("Check 8.")

            if isinstance(answer, dict):
                asyncio.create_task(generate_kohana_voice_async("Here you go!"))
                return answer

            if answer and answer != "Sorry, I’m stuck.":
                asyncio.create_task(generate_kohana_voice_async(answer))
                return {"text": answer, "type": "text"}

        print("Check 9.")

        user_profile = load_user_profile_by_id(user_id)
        context = (
            f"### Instruction: \n"
            f"{kohana_personality}\n"
            f"<meta>{meta_json}</meta>\n"
            "You can *reference* the fields inside <meta>, but do **NOT** quote the JSON itself.\n"
            "Respond only in natural language as Yuki Kohana.\n\n"
            f"Conversation so far:\n{history}\n"
            f"### Input: \n {user_profile['name']}: {prompt}\n ### Response: \n Kohana:"
        )
        print("Check 10.")
        decoded = llm_generate(context)
        print("===================================================================")
        print(decoded)
        print("===================================================================")

        response = decoded.split("Kohana:")[-1].strip() if "Kohana:" in decoded else decoded.strip()
        if f"{user_profile['name']}:" in response:
            response = response.split(f"{user_profile['name']}:")[0].strip()
        if not response or len(response.strip()) < 2:
            response = "I'm still here, I just need a moment. Could you ask that again?"

        memory_module.add_memory(f"{user_profile['name']}: {prompt}\nKohana: {response}")      
        asyncio.create_task(generate_kohana_voice_async(response))
        time.sleep(.5)
        print(generate_kohana_reactions(response))

        formatted = format_for_display(response)
        conversation_state["formatted_response"] = formatted
        return formatted

    except Exception as e:
        print(f"Error in generate_response: {e}")
        if retries < MAX_RETRIES:
            return generate_response(prompt, history, username, max_length, conversation_state, retries=retries + 1, scratch_used=True)
        return "Sorry, something went wrong."

    finally:
        gc.collect()

def format_for_display(response: str) -> dict:
    """Detect code or list patterns and return structured content."""
    is_code = bool(re.search(r"(def |class |for |while |if |else:|try:|except|import |\{|\})", response)) and "\n" in response
    is_list = bool(re.search(r"^\s*[\-\*\d]+\.", response, re.MULTILINE)) or bool(re.search(r"^\s*[\-\*]\s", response, re.MULTILINE))
    
    return {
        "text": response.strip(),
        "type": "code" if is_code else "list" if is_list else "text"
    }


def trigger_action(intent_type, prompt, conversation_state):
    INTENT_HANDLERS = {
        "take_note":    handle_note_interaction,
        "send_email":   handle_email_interaction,
        "set_reminder": handle_reminder_interaction,
        "set_timer":    handle_timer_interaction,
        "search_web":   handle_web_search,
    }      
    handler = INTENT_HANDLERS.get(intent_type)
    if handler:
        response = handler(prompt, conversation_state)
        generate_kohana_voice_async(response)
        print(f"✅ Triggered intent: {intent_type}")
        return response
    else:
        print(f"⚠️ No handler found for intent: {intent_type}")
        return None

def intent_checker(prompt: str, intention: str) -> str:
    if not intention:
        return "no"
    q = (
        f'User: "{prompt}"\n'
        f'Does the user explicitly ask to **{intention}**? '
        f"Respond ONLY yes or no."
    )
    ans = llm_generate(q, max_new_tokens=2).lower()
    ans = "yes" if ans.strip().startswith("y") else "no"
    print("🔎 Intent-checker:", ans)
    return ans

def start_conversation():
    local_memory = MemoryModule(debug=True)
    prompt_seed = random.choice([
        "What have I learned today?",
        "What emotion stands out right now?",
        "Where did I help (or hinder) Malachi?",
        "What surprised me recently?",
    ])
    thoughts = local_memory.retrieve_relevant_memories(prompt_seed,
                                                    context="personal_thoughts",
                                                    top_k=5)
    return thoughts

def save_conversation(log_data, log_dir=LOG_DIR):
    log_file = os.path.join(log_dir, f"conversation_{len(os.listdir(log_dir)) + 1}.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"Conversation saved to {log_file}.")

def manage_kitsune_bucks(action, amount):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "kitsune_bucks.db")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kitsune_bucks (
            id INTEGER PRIMARY KEY,
            total REAL
        )
    """)
    cursor.execute("SELECT total FROM kitsune_bucks WHERE id = 1")
    row = cursor.fetchone()
    if row:
        total_bucks = row[0]
    else:
        total_bucks = 0.0
        cursor.execute("INSERT INTO kitsune_bucks (id, total) VALUES (1, ?)", (total_bucks,))

    if action == "add":
        total_bucks += amount
    elif action == "remove":
        total_bucks = max(0, total_bucks - amount)

    cursor.execute("UPDATE kitsune_bucks SET total = ? WHERE id = 1", (total_bucks,))
    conn.commit()
    conn.close()
    return total_bucks

def get_kitsune_bucks():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "kitsune_bucks.db")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kitsune_bucks (
            id INTEGER PRIMARY KEY,
            total REAL
        )
    """)
    cursor.execute("SELECT total FROM kitsune_bucks WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0.0

def needs_reasoning(prompt: str) -> bool:
    trivial = any(p in prompt.lower()
                  for p in ("hi", "hello", "how are you", "good night"))
    return not trivial

def build_meta_json(
        user_id: str,
        goals: str,                  # <─ was str | list, keep it simple
        detected_intent: str | None,
        conversation_state: dict
) -> str:
    """
    Build the lightweight meta-object that will be injected into
    Kohana’s scratch-pad prompt.

    Feel free to add / remove fields; just keep it small.
    """
    meta = {
        "user": user_id,
        "goals": goals,                              # single sentence is fine
        "kitsune_bucks": get_kitsune_bucks(),
        "intent": detected_intent or "none",
        "last_image": conversation_state.get("last_user_image"),
        "pending_steps": {
            k: v for k, v in conversation_state.items()
            if k.endswith("_step")                   # e.g. reminder_step
        },
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    # one-liner JSON; ensure_ascii=False keeps it readable
    return json.dumps(meta, separators=(",", ":"), ensure_ascii=False)

# Assistant Funtions
# -- Tool Registry ----------------------------------------------------

INTENT_KEYWORDS.update({          # add any new keywords you like
    "weather": ["weather", "forecast"],
    "calc": ["calculate", "calc", "what is", "=", "plus", "minus", "times",
             "divided by", "percentage", "sqrt", "^", "power of"]
})

_SAFE_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "pow": pow, "round": round,
    # add more if you need them
}

def run_tool(name, arg):
    if name == "calc":
        try:
            return _safe_eval(arg)
        except Exception as e:
            return f"error: {e}"
    if name == "search_web":
        return web_search(arg)           # returns short snippet
    if name == "memory":
        return memory_module.retrieve(arg)
    if name == "note":
        save_note(*arg.split("|",1)); return "saved"
    if name == "gen_image":
        from image_generator_module import tool_entry
        return tool_entry(arg) 
    if name == "refine_image":       # ← model refines last user upload
        from image_generator_module import refine_existing
        return {"type": "image",
                "text": refine_existing(arg["path"], arg.get("instructions",""))}
    # add more…
    return "unknown tool"

MAX_PASSES = 3
SCRATCH_TEMPLATE = """
You are Yuki Kohana. You have a scratchpad to help you think step-by-step.

If you need a tool, output JSON like this: 
{{"tool":"name", "arg":"value"}} 

If youre still thinking, JSON like this: 
{{"thought":"…", "final":false}}

When you’re ready to reply to the user, output JSON: 
{{"final":true, "answer":"…"}}

META: {meta_json}
USER: "{user_msg}"

Rules:
- Always format output as a SINGLE valid JSON object
- "answer" must ONLY appear if "final": true
- If final is true, "answer" is REQUIRED
- Never include explanations outside of the JSON
- Output NOTHING except valid JSON

Begin below:
"""

def scratch_loop(user_msg: str, meta_json: str) -> tuple[str, list[str]]:
    print("Entered Scratch Loop")
    scratch, thoughts = [], []

    for pass_num in range(MAX_PASSES):
        print(f"🔁 Scratch pass #{pass_num + 1}")
        prompt = SCRATCH_TEMPLATE.format(meta_json=meta_json, user_msg=user_msg) + "\n\n".join(scratch)
        raw = llm_generate(prompt)
        print("Raw Scratch Output:", raw)

        try:
            obj = json.loads(raw.splitlines()[-1])
            print("✅ JSON parsed successfully.")
        except json.JSONDecodeError:
            print("❌ JSON parse failed.")
            continue  # Try again with next loop iteration

        scratch.append(raw)

        if "thought" in obj:
            thoughts.append(obj["thought"])

        if "tool" in obj:
            obs = run_tool(obj["tool"], obj["arg"])
            scratch.append(json.dumps({"observation": obs}))
        elif obj.get("final"):
            return obj["answer"], thoughts

    print("❌ Max passes exhausted or malformed output.")
    return "Sorry, I’m stuck.", thoughts


def _safe_eval(expr: str) -> str:
    """
    Evaluate a basic arithmetic expression safely.
    Allowed nodes: Constant, BinOp (+-*/%**), UnaryOp (+/-), Call (sqrt/…).
    """
    allowed_bin = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):          # numbers
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed_bin[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):           # +3 / -2
            return +_eval(node.operand) if isinstance(node.op, ast.UAdd) else -_eval(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _SAFE_FUNCS.get(node.func.id)
            return fn(*[_eval(arg) for arg in node.args]) if fn else None
        raise ValueError("unsafe")
    tree = ast.parse(expr, mode="eval")
    return str(_eval(tree.body))

# very tiny helpers so run_tool doesn’t throw
def web_search(query: str) -> str:
    webbrowser.open(f"https://www.google.com/search?q={query}")
    return f"opened browser for: {query}"


def memory_search(query: str) -> str:
    results = memory_module.retrieve_relevant_memories(query, top_k=2)
    return results[0]["content"] if results else "no memory found"

def save_note(title, content):
    note_path = NOTES_DIR / f"{title}.txt"
    with open(note_path, "w") as file:
        file.write(content)
    print(f"Note '{title}' saved!")

def read_note(title):
    note_path = NOTES_DIR / f"{title}.txt"
    if note_path.exists():
        with open(note_path, "r") as file:
            return file.read()
    else:
        return f"Note '{title}' not found!"

def list_notes():
    return [note.stem for note in NOTES_DIR.iterdir() if note.is_file()]

def handle_note_interaction(prompt, conversation_state):
    step = conversation_state.get("note_step")
    note_data = conversation_state.get("note_data", {})

    if not prompt.strip():
        return "Sure! What would you like to title your note?"

    if step == "awaiting_title":
        note_data["title"] = prompt.strip()
        conversation_state["note_step"] = "awaiting_content"
        conversation_state["note_data"] = note_data
        return f"Got it. What should the note titled '{note_data['title']}' say?"

    elif step == "awaiting_content":
        note_data["content"] = prompt.strip()
        save_note(note_data["title"], note_data["content"])
        conversation_state.pop("note_step", None)
        conversation_state.pop("note_data", None)
        return f"Done! I’ve saved the note titled '{note_data['title']}'."

    return "I'm not sure how to continue with the note. Let's try again."


def send_email_with_attachment(to_email, subject, message, attachment_path=None, from_email="scarletrunner2.0@gmail.com", password="fgfa gnyw ngme mkrz"):
    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(message)

    if attachment_path:
        attachment_file = Path(attachment_path)
        with open(attachment_file, "rb") as f:
            file_data = f.read()
            file_name = attachment_file.name
            maintype, subtype = "application", "octet-stream"

            if attachment_file.suffix in [".jpg", ".jpeg"]:
                maintype, subtype = "image", "jpeg"
            elif attachment_file.suffix == ".png":
                maintype, subtype = "image", "png"
            elif attachment_file.suffix == ".gif":
                maintype, subtype = "image", "gif"

            msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)

    print(f"Email with attachment sent to {to_email}!")

def handle_email_interaction(prompt, conversation_state):
    step = conversation_state.get("email_step")
    email_data = conversation_state.get("email_data", {})

    if not prompt.strip():
        return "Sure! Who should I send the email to?"

    if step == "awaiting_recipient":
        email_data["to"] = prompt.strip()
        conversation_state["email_step"] = "awaiting_subject"
        conversation_state["email_data"] = email_data
        return "What should the subject line be?"

    elif step == "awaiting_subject":
        email_data["subject"] = prompt.strip()
        conversation_state["email_step"] = "awaiting_content"
        conversation_state["email_data"] = email_data
        return "What should the message say?"

    elif step == "awaiting_content":
        email_data["content"] = prompt.strip()
        conversation_state["email_step"] = "awaiting_attachment"
        conversation_state["email_data"] = email_data
        return "Would you like to attach a file? If yes, please provide the path or say 'no'."

    elif step == "awaiting_attachment":
        attachment = prompt.strip()
        email_data["attachment"] = None if attachment.lower() == "no" else attachment
        send_email_with_attachment(
            email_data["to"],
            email_data["subject"],
            email_data["content"],
            email_data.get("attachment")
        )
        conversation_state.pop("email_step", None)
        conversation_state.pop("email_data", None)
        return f"Email sent to {email_data['to']}!"

    return "I'm not sure how to continue with the email. Let's try again."

def handle_reminder_interaction(prompt, conversation_state):
    step = conversation_state.get("reminder_step")
    reminder_data = conversation_state.get("reminder_data", {})

    print(f"➡️ [Reminder Handler] Entered with step: {step}")
    print(f"➡️ Prompt: {prompt.strip()}")

    if not prompt.strip():
        return "Alright, when should I remind you? (Please specify a date/time or something like 'in 10 minutes'.)"

    if step == "awaiting_time":
        try:
            user_text = prompt.lower()
            now = datetime.datetime.now()

            if "second" in user_text:
                num = int(user_text.split("second")[0].split()[-1])
                remind_time = now + datetime.timedelta(seconds=num)
            elif "minute" in user_text:
                num = int(user_text.split("minute")[0].split()[-1])
                remind_time = now + datetime.timedelta(minutes=num)
            elif "hour" in user_text:
                num = int(user_text.split("hour")[0].split()[-1])
                remind_time = now + datetime.timedelta(hours=num)
            else:
                remind_time = dateutil.parser.parse(prompt, fuzzy=True)
                if remind_time < now:
                    return "The time you gave is in the past. Could you specify a future time?"

            print(f"✅ Parsed reminder time: {remind_time}")

            reminder_data["time"] = remind_time
            conversation_state["reminder_step"] = "awaiting_message"
            conversation_state["reminder_data"] = reminder_data

            return "Great! What is the reminder message?"

        except Exception as e:
            print(f"⚠️ Reminder time parsing failed: {e}")
            conversation_state.pop("reminder_step", None)
            conversation_state.pop("reminder_data", None)
            return "I'm sorry, I couldn't understand the reminder time. Could you try again?"

    elif step == "awaiting_message":
        reminder_data["message"] = prompt.strip()
        print(f"✅ Setting reminder with time: {reminder_data['time']}, message: {reminder_data['message']}")

        conversation_state.setdefault("reminders", []).append(reminder_data)
        schedule_reminder(reminder_data["time"], reminder_data["message"])
        conversation_state.pop("reminder_step", None)
        conversation_state.pop("reminder_data", None)

        return f"Reminder set for {reminder_data['time']}. I'll remind you: \"{reminder_data['message']}\"."

    # Fallback if somehow the state is invalid
    print("⚠️ Reminder state invalid or unknown. Resetting.")
    conversation_state.pop("reminder_step", None)
    conversation_state.pop("reminder_data", None)
    return "I got a little mixed up while setting that reminder. Let's start again — when should I remind you?"


def load_scheduled_events():
    if EVENTS_FILE.exists():
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    else:
        return []

def save_scheduled_events(events):
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

def add_scheduled_event(event_type, trigger_time, message):
    events = load_scheduled_events()
    events.append({
        "type": event_type,
        "trigger_time": trigger_time.isoformat(),
        "message": message
    })
    save_scheduled_events(events)

def remove_scheduled_event(event_type, trigger_time, message):
    events = load_scheduled_events()
    updated = [
        e for e in events
        if not (e["type"] == event_type and e["trigger_time"] == trigger_time.isoformat() and e["message"] == message)
    ]
    save_scheduled_events(updated)

def restore_scheduled_events():
    events = load_scheduled_events()
    now = datetime.datetime.now()

    for event in events:
        event_time = datetime.datetime.fromisoformat(event["trigger_time"])
        if event_time > now:
            if event["type"] == "reminder":
                schedule_reminder(event_time, event["message"])
            # (Later you could add: elif event["type"] == "timer": schedule_timer(...))


def schedule_reminder(remind_time, message):
    add_scheduled_event("reminder", remind_time, message)  # <--- SAVE IT!

    def reminder_thread():
        now = datetime.datetime.now()
        wait_seconds = (remind_time - now).total_seconds()
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        print(f"\n[REMINDER] {message}\n")
        subject = "🦊✨Reminder From Kohana!"
        body = f"Hey! This is your friendly reminder:\n\n{message}\n\nSet for: {remind_time.strftime('%Y-%m-%d %H:%M:%S')}"
        assistant_email(subject, body)
        remove_scheduled_event("reminder", remind_time, message)  # <--- REMOVE IT AFTER FIRING!

    t = threading.Thread(target=reminder_thread, daemon=True)
    t.start()


def handle_timer_interaction(prompt, conversation_state):
    step = conversation_state.get("timer_step")
    timer_data = conversation_state.get("timer_data", {})

    if not prompt.strip():
        return "How long should the timer be? (e.g., 30 seconds, 5 minutes)"

    if step == "awaiting_duration":
        try:
            user_text = prompt.lower()
            if "minute" in user_text:
                num_val = int(user_text.split("minute")[0].split()[-1])
                duration_seconds = num_val * 60
            elif "second" in user_text:
                num_val = int(user_text.split("second")[0].split()[-1])
                duration_seconds = num_val
            else:
                # fallback if unclear
                duration_seconds = 30

            timer_data["duration"] = duration_seconds
            conversation_state.pop("timer_step", None)
            conversation_state.pop("timer_data", None)

            schedule_timer(duration_seconds)
            return f"Timer set for {duration_seconds} seconds."

        except Exception:
            conversation_state.pop("timer_step", None)
            conversation_state.pop("timer_data", None)
            return "I couldn't understand that time. Let's try again."

    return "I'm not sure how to proceed with the timer."

def schedule_timer(duration_seconds):
    def timer_thread():
        time.sleep(duration_seconds)
        print(f"\n[TIMER COMPLETED] Your {duration_seconds}-second timer just finished!\n")
        subject = "🦊Timer is up!✨"
        body = f"Hey! Your timer for {duration_seconds} seconds has completed!\n"
        assistant_email(subject, body)
    t = threading.Thread(target=timer_thread, daemon=True)
    t.start()

import webbrowser

def handle_web_search(prompt, conversation_state):
    """
    Handles a web search by extracting a query from the prompt and opening it in the browser.
    Prevents re-entry by using conversation state.
    """
    if conversation_state.get("search_in_progress"):
        return "I'm already working on that search—give me a moment!"

    # Mark search in progress
    conversation_state["search_in_progress"] = True

    try:
        # Naive keyword removal (you may want regex or NLP later)
        query = prompt.lower().replace("search", "").replace("google", "").strip()
        query = query.replace("for", "").strip()

        if query:
            webbrowser.open(f"https://www.google.com/search?q={query}")
            response = f"Okay, searching the web for: {query}"
        else:
            response = "I didn’t catch what to search. Could you repeat that?"

    except Exception as e:
        response = f"Error while searching: {e}"

    # Reset flag to prevent loop on next round
    conversation_state["search_in_progress"] = False
    return response
