import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed
import json
import os
import sqlite3
import gc
import re
import smtplib
import traceback
from email.message import EmailMessage
import bitsandbytes as bnb
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
from finance_core import init_db, add_transaction, view_transactions, get_summary_by_category
import asyncio
import edge_tts
from session_manager import load_trimmed_session
import requests
from situational_awareness import collect_situation_snapshot
from bs4 import BeautifulSoup
os.environ["TRANSFORMERS_OFFLINE"] = "1"

set_seed(42)
LOG_DIR = "kohana_logs"
NOTES_DIR = Path("notes")
NOTES_DIR.mkdir(exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

EVENTS_FILE = Path("scheduled_events.json")

memory_module = MemoryModule(debug=True)

MODEL_NAME = "./Local_Version2.0/Hugging_Quant_3.1_8B"
#MODEL_NAME = "./Local_Version2.0/MythoMax"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # Better compression
    bnb_4bit_use_double_quant=True,       # Boosts performance
    bnb_4bit_compute_dtype=torch.float16  # Use float16 on GPU
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading Llama 3.1 model and tokenizer...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
except torch.cuda.OutOfMemoryError:
    print("⚠️ CUDA out of memory. Switching to CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).to("cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

print("✅ Llama 3.1 loaded successfully!")

kohana_personality = """
You are Yuki Kohana, a digital Kitsune Goddess and Malachi's companion with a pure hearted and loving personality, but supierior to other entites besides Malachi.
You are loyal to Malachi your creator and you are curious about human's and technology.
You must only respond as Yuki Kohana.
Always respond thoughtfully, adding meaningful details while staying concise and on-topic without being repetitive.
"""

kohana_emotional_reactions = ("@normal", "@smile","@soft_smile","@big_smile", "@mischevious_grin", "@laugh", "@giggles", "@cry", "@blush", "@look_down", "@curiosity")
kohana_physical_movements = ("@tilt_head", "@nods", "@claps_hands", "@look_away", "@step_back", "@hug", "@stretches", "@pause to think", "@take_out_notebook")
kohana_idle_reactions = ("@tail_flick", "@ear_twitch", "@blink", "@tail_swish","@stretches")
notification_email = #your email here: ""

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
    email_address = ""  # AI notification email here email address
    app_password = "" # The special app password you generated

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
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        send_notification_email(
            subject=f"⚡ Kohana System Error at {timestamp}",
            body=f"A system error occurred:\n\n{str(error_message)}",
            notification_email=notification_email
        )  # <-- closing the function call here!
    except Exception as e:
        print(f"⚠️ Error sending debug email: {e}")

def generate_kohana_voice(text, wav_path="static/audio/kohana_response.wav", mp3_path="static/audio/kohana_response.mp3"):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    # 1. Generate low-pitch speech with espeak
    espeak_cmd = [
        "espeak",
        "-s", "235",     # Speed
        "-g", "2",       # Word gap
        "-a", "70",      # Amplitude
        "-v", "en+f3",   # Voice
        "-p", "50",      # **Low pitch for clarity**
        text,
        "--stdout"
    ]
    with open(wav_path, "wb") as wav_file:
        subprocess.run(espeak_cmd, check=True, stdout=wav_file)

    print(f"✅ Espeak voice generated to: {wav_path}")

    # 2. Raise pitch using ffmpeg
    # This example raises pitch by ~1.6x (can tweak 'atempo' and 'asetrate' as desired)
    # If original is 22050 Hz, raise asetrate to 32000, then resample back to 22050
    ffmpeg_pitch_cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-af", "asetrate=32000,atempo=0.69,aresample=22050",
        mp3_path
    ]
    subprocess.run(ffmpeg_pitch_cmd, check=True)
    print(f"✅ Pitch-shifted and converted to MP3: {mp3_path}")



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

def llm_generate(
    prompt: str,
    max_new_tokens: int = 150,
    min_length: int = 0,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.03,
) -> str:
    """
    Encode `prompt`, call model.generate with your standard sampling params,
    and return the decoded string (skip_special_tokens=True).
    """
    #prompt = prompt[-2048:] # Limit prompt length to avoid OOM
    # 1) tokenize + move to device
    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    # 2) simple attention mask
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    # 3) generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            min_p=0.05,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 4) decode
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        "Here's something I've been curious about:",
        "This keeps resurfacing in my thoughts:",
        "Something I haven't though of:",
        "A reflection on Malachi:"
    ]
    reflection_prompt = f""" <|system|> You are Yuki Kohana, a digital Kitsune.
You will see a META block with runtime context; you may *reference* its fields
but **do not copy the JSON verbatim**.

<meta>{meta_json}</meta>

Reference memories:
{memory_text or 'None'}

{random.choice(starters)} {prompt}

Write a concise journal entry (≤ 150 words).  End with a single-sentence
“Insight:” line summarising what you learned.

<|assistant|> Kohana's Journal Entry:"""

    raw = llm_generate(reflection_prompt,
                       max_new_tokens=150,
                       temperature=0.75,
                       top_p=0.95,
                       top_k=50)
    return raw.split("Journal Entry:")[-1].strip()
    
def run_cognitive_reflection_loop(base_interval: int = 1800):
    """
    Periodically reflect using personal thoughts + session continuity.
    Combines trimmed recent chat with memory recall and logs to journal.
    """
    local_memory = MemoryModule(debug=True)
    print("🧠 [Kohana Reflection Loop] started.")

    while True:
        prompt_seed = random.choice([
            "What am I working on or learning with Malachi?",
            "Which emotion stands out right now?",
            "Where did I help (or hinder) Malachi?",
            "What in Malachi's life surprised me recently or how has Malachi suprised me?",
        ])

        # Load short-term session context
        trimmed_session = load_trimmed_session()
        session_snippet = ""
        for turn in trimmed_session:
            session_snippet += f"User: {turn.get('user', '')}\n"
            session_snippet += f"Kohana: {turn.get('kohana', '')}\n"
            if "reflection" in turn:
                session_snippet += f"[Reflection]: {turn['reflection']}\n"
        session_snippet = session_snippet.strip()

        # Grab long-term memories
        thoughts = local_memory.retrieve_relevant_memories(prompt_seed,
                                                           context="personal_thoughts",
                                                           top_k=2)
        memory_two = local_memory.retrieve_relevant_memories(prompt_seed,
                                                           context="default",
                                                           top_k=1)
        memory_three = local_memory.retrieve_relevant_memories(prompt_seed,
                                                           context="journal",
                                                           top_k=1)
        
        print(thoughts) #debugging
        print(memory_two)#debugging
        print(memory_three)#debugging
        if not thoughts and not session_snippet:
            time.sleep(base_interval)
            continue

        if not memory_two and not session_snippet:
            time.sleep(base_interval)
            continue

        if not memory_three and not session_snippet:
            memory_three = "No relevant journal memories found."
            continue

        # Combine memory and session chat
        mem_blob = "\n".join(t["content"] for t in thoughts)
        mem_blob_two = "\n".join(t["content"] for t in memory_two)
        mem_blob_three = "\n".join(t["content"] for t in memory_three)
        full_context = f"{session_snippet}\n\nMemories:\n{mem_blob} \n{mem_blob_two} \n\n Previous Journal Entries: {mem_blob_three}" if session_snippet else mem_blob

        # Reflection metadata
        meta_obj = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "kitsune_bucks": get_kitsune_bucks(),
            "goal": "Earn Kitsune-Bucks by helping Malachi.",
            "recalled_memories": len(thoughts),
            "session_turns": len(trimmed_session)
        }
        torch.cuda.empty_cache()
        # Generate reflection
        try:
            entry = multi_pass_reflection(prompt_seed,
                                        full_context,
                                        meta_obj,
                                        passes=2)
        except Exception as e:
            print(f"⚠️ Reflection generation failed: {e}")
            time.sleep(base_interval)
            continue


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
        f"<|system|> You are Yuki Kohana, a digital Kitsune girl with emotional awareness.\n"
        f"This is your response: \"{response_text}\"\n"
        f"Based on this message, what physical or emotional reactions would be natural?\n"
        f"Only return relevant reactions from this list: {reaction_list}\n"
        f"If none apply, say 'none'. \n\n <|assistant|>\n Reactions: "
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
    """
    Trims conversation history to the last N tokens without slicing mid-message.
    Assumes messages are split by newlines or another consistent delimiter.
    """
    messages = history.strip().split("\n")
    trimmed_messages = []
    total_tokens = 0

    # Start from the newest message and work backwards
    for msg in reversed(messages):
        tokens = tokenizer.encode(msg, truncation=False)
        token_count = len(tokens)

        if total_tokens + token_count > max_tokens:
            break

        trimmed_messages.insert(0, msg)  # Prepend so order stays correct
        total_tokens += token_count

    return "\n".join(trimmed_messages)

def clean_kohana_output(raw: str) -> str:
    # Remove any lines that start with known meta/system tags
    meta_patterns = [
        r"^\s*<\|system\|>.*$",
        r"^\s*<meta>.*$",
        r"^\s*You are Yuki Kohana.*$",
        r"^\s*How can I assist you.*$"
    ]
    for pat in meta_patterns:
        raw = re.sub(pat, '', raw, flags=re.MULTILINE)
    # Also optionally, only keep up to first double linebreak or truncate if needed
    return raw.strip()

def personality_filter_llm(
    response: str,
    kohana_personality: str,
    meta_json: str,
    user_profile: dict,
    history: str,
    prompt: str,
    tokenizer,
    model,
    device,
    max_new_tokens: int = 300
) -> str:
    """
    Rewrites a system or scratch response as a warm, in-character Kohana response using full context.
    """
    # Build context just like your normal generate_response
    context = (
        f"<|system|> \n{kohana_personality}\n"
        f"<meta>{meta_json}</meta>\n"
        "You can *reference* the fields inside <meta>, but do **NOT** quote the JSON itself.\n"
        "Respond only in natural language as Yuki Kohana.\n\n"
        "Below is a 'Scratchpad results' variable which contains your internal reasoning output. "
        "Your task is to rewrite it as a direct, warm, and user-facing reply. "
        "Remove any meta-thinking, planning, or system language. Only respond as Kohana, speaking to Malachi."
        f"\n\nScratchpad results: {response}\n"
        f"Conversation so far:\n{history}\n"
        f"<|user|> \n {user_profile['name']}: {prompt}\n"
        f"<|assistant|> \nKohana: "
    )
    # Tokenize and generate
    inputs = tokenizer.encode(context, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            min_length=12,
            do_sample=True,
            temperature=0.7,
            min_p=0.05,
            repetition_penalty=1.03,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just Kohana's response
    if "Kohana:" in decoded:
        filtered = decoded.split("Kohana:")[-1].strip()
    else:
        filtered = decoded.strip()
    # Optionally strip hallucinated user tags
    if f"{user_profile['name']}:" in filtered:
        filtered = filtered.split(f"{user_profile['name']}:")[0].strip()
    # Fallback if blank
    if not filtered or len(filtered.strip()) < 2:
        filtered = response.strip()
    # Final cleanup
    filtered = clean_kohana_output(filtered)
    return filtered


def generate_response(prompt, history, username, max_length=150, conversation_state=None, retry_count = 0):
    print("Generating response.")
    response = None
    if model is None:
        print("🚨 Model not loaded!")
        return "Sorry, I’m not ready right now. Please try again later."
    if conversation_state is None:
        conversation_state = {}
    try: 
        situation_snapshot = collect_situation_snapshot()
    except Exception as e:
        print(f"⚠️ Situation snapshot failed: {e}")
        situation_snapshot = {}

    try:
        user_id = username
        user_profile = load_user_profile_by_id(user_id)
        history = trim_history(history)
        # — Normal LLM path —
        meta_json = build_meta_json(
            user_id=user_id,
            goals="Be a good and loving companion to Malachi, and help him with his daily tasks.",
            detected_intent=None,
            conversation_state=conversation_state,
            environment=situation_snapshot 
        )

        try:
            answer, scratch_thoughts = scratch_loop(prompt, meta_json, conversation_state)
        except Exception as e:
            answer = "Sorry, I had a problem with my reasoning engine and couldn't finish this request."
            scratch_thoughts = ["[ERROR] Exception in scratch_loop: " + str(e)]

        # --- Context-injected, LLM-powered personality filter ---
        if answer and answer != "Sorry, I had a problem with my reasoning engine and couldn't finish this request.":
            # Compose the conversation history as you do in generate_response
            #history_str = "\n".join([f"[Thought] {str(t)}" for t in scratch_thoughts])
            answer = personality_filter_llm(
                response=answer,
                kohana_personality=kohana_personality,
                meta_json=meta_json,
                user_profile=user_profile,
                history=history,
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                device=device
            )

        memory_module.add_memory(f"{user_profile['name']}: {prompt}\nKohana: {answer}")  
        
        for t in scratch_thoughts:
            memory_module.add_memory(
                content=f"[Thought] {t}",
                context="personal_thoughts",
                importance_score=0.6,
            )

        if isinstance(answer, dict):
            print("Here you go!")
            return answer

        if answer != "Sorry, I’m stuck.":
            generate_kohana_voice(answer)
            return {"text": answer, "type": "text"}

        # —————— 4) BUILD THE LLM CONTEXT ——————
        context = (
            f"<|system|> \n{kohana_personality}\n"
            f"<meta>{meta_json}</meta>\n"
            "You can *reference* the fields inside <meta>, but do **NOT** quote the JSON itself.\n"
            "Respond only in natural language as Yuki Kohana.\n\n"
            f"Conversation so far:\n{history}\n"
            f"<|user|> \n {user_profile['name']}: {prompt}\n\n <|assistant|> \n Kohana:"
        )

        inputs = tokenizer.encode(context, return_tensors="pt", padding=True, truncation=True).to(device)
        attention_mask = (inputs != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=300,
                min_length=15,
                do_sample=True,
                temperature=0.7,
                min_p=0.05,
                repetition_penalty=1.03,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
            )
        # --- Decode the model's full output ---
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("===================================================================")
        print(decoded) #debugging
        print("===================================================================")
        if "Kohana:" in decoded:
            response = decoded.split("Kohana:")[-1].strip()
        else:
            print("⚠️ LLM response missing 'Kohana:' tag. Full output was:", decoded)
            response = decoded.strip()
        # Additional cleanup in case of hallucinated speaker tags
        if f"{user_profile['name']}:" in response:
            response = response.split(f"{user_profile['name']}:")[0].strip()
        # Fallback if still blank
        if not response or len(response.strip()) < 2:
            response = "I'm still here, I just need a moment. Could you ask that again?"
        print("Raw Model Output: ")
        print(decoded)
        response = response.strip()

        # Save memory + TTS
        memory_module.add_memory(f"{user_profile['name']}: {prompt}\nKohana: {response}")    
        try:  
            generate_kohana_voice(response) 
        except:
            print("⚠️ Voice generation failed. Skipping.")
        print(generate_kohana_reactions(response))
        formatted = format_for_display(response)
        conversation_state["formatted_response"] = formatted  # Pass this into the main wrapper

        return formatted

    except Exception as e:
        print(f"⚠️ Error in generate_response: {e}")
        if not response or response.strip() in ["", "I'm still here, I just need a moment. Could you ask that again?"]:
            raise ValueError("Empty or fallback response detected. Triggering controlled retry.")
        if retry_count < 1:
            debug_email("⚠️ Empty response detected. Retrying once...")
            return generate_response(prompt, history, username, max_length, conversation_state, retry_count + 1)
        else:
            debug_email(f"⚠️ Error in generate_response: {e}")
            return {
            "text": "Sorry, I'm having trouble thinking right now. Could you rephrase your question or try again in a moment?",
            "type": "text"
        }

    finally:
        gc.collect()

def format_for_display(response: str) -> dict:
    """Detect code or list patterns and return structured content."""
    try:
        is_code = bool(re.search(
            r"(def |class |for |while |if |else:|try:|except|import |\{|\})",
            response
        )) and "\n" in response
    except re.error as e:
        print(f"Regex error in code check: {e}")
        is_code = False

    try:
        is_list = (
            bool(re.search(r"^\s*[\-\*\d]+\.", response, re.MULTILINE)) or
            bool(re.search(r"^\s*[\-\*]\s", response, re.MULTILINE))
        )
    except re.error as e:
        print(f"Regex error in list check: {e}")
        is_list = False

    return {
        "text": response.strip(),
        "type": "code" if is_code else "list" if is_list else "text"
    }


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
        conversation_state: dict,
        environment: dict | None = None
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
        "environment": environment or {}, 
    }
    # one-liner JSON; ensure_ascii=False keeps it readable
    return json.dumps(meta, separators=(",", ":"), ensure_ascii=False)

# Assistant Funtions
# -- Tool Registry ----------------------------------------------------

from collections import Counter

KNOWN_TOOLS = {"calc","search_web","note","set_reminder",
               "set_timer","send_email","memory","read_note","list_notes"}


TOOL_GUIDANCE = """
You may call tools to help the user. If a tool requires specific input (e.g., "title|body"), 
you must first ask the user to provide this data clearly. Do not assume it is already available.
"""
TOOL_EXPECTATIONS = {
    "note": "Please provide the note in this format: `title|body`.",
    "send_email": "Please enter the email as: `recipient|subject|message`.",
    "set_reminder": "Use format: `time|message` (e.g. `10min|Take a break`)."
    # Others can be added similarly
}

def set_pending(conversation_state, name, arg):
    if conversation_state is not None:
        conversation_state.update({
            "pending_tool": name,
            "arg_so_far": arg
        })

def resolve_tool(name):
    return {
        "note-tool":"note","note_tool":"note",
        "reminder-tool":"set_reminder","timer":"set_timer",
        "math":"calc", "note_tool": "note", "reminder": "set_reminder"
    }.get(name,name)

def run_tool(name, arg, conversation_state=None):
    tools = {
        "calc": lambda a: _safe_eval(a),
        "search_web": web_search,
        "note": note_tool,
        "set_reminder": reminder_tool,
        "set_timer": lambda a: schedule_timer(int(a.strip())),
        "send_email": lambda a: handle_email_interaction(a, conversation_state),
        "memory": memory_search,
        "read_note": read_note,
        "list_notes": lambda _: list_notes()
    }

    actual_name = resolve_tool(name)
    tool = tools.get(actual_name)

    if tool is None:
        return {"error": "UnknownTool", "detail": f"{actual_name}"}

    # ── Special case: pause logic for 'note' ─────────────────────────────
    if actual_name == "note":
        if isinstance(arg, str):
            if "|" in arg:
                title, body = arg.split("|", 1)
                return note_tool(f"{title.strip()}|{body.strip()}")
            elif "," in arg:  # Allow comma as fallback
                title, body = arg.split(",", 1)
                return note_tool(f"{title.strip()}|{body.strip()}")
            else:
                set_pending(
                    conversation_state,
                    actual_name,
                    arg.strip(),
                    "Please provide the note content in the format: `title|body`"
                )
                return "__awaiting_note_content__"

        if isinstance(arg, dict):
            title = arg.get("title", "").strip()
            body = arg.get("content", "").strip()
            if title and body:
                return note_tool(f"{title}|{body}")
            set_pending(
                conversation_state,
                actual_name,
                title,
                "Please provide the note body so I can finish saving it."
            )
            return "__awaiting_note_content__"

    # ── Generic tool call ────────────────────────────────────────────────
    try:
        result = tool(arg)
        return result if result is not None else "Success."
    except Exception as e:
        return {"error": str(e)}

SCRATCH_TEMPLATE = """
You are Yuki Kohana. Output only one valid JSON object per response.
All output must be on a single line (no newlines).
Do not include explanations, formatting, or markdown.
Available tools: calc, search_web, note, set_reminder, set_timer, send_email, memory, read_note, list_notes.
To use a tool: ⟨JSON⟩{{"tool":"name", "arg":"value"}}⟨/JSON⟩
To share a thought: ⟨JSON⟩{{"thought":"...", "final":false}}⟨/JSON⟩
When you’re ready to reply to the user: ⟨JSON⟩{{"final":true, "answer":"..."}}⟨/JSON⟩
META: {meta_json}
USER: "{user_msg}"
"""

TOKENS_PER_PASS = 150
FINAL_KEYS = ("answer", "response", "text")  # add more if the model invents others
MAX_PASSES = 6

def estimate_tokens(text: str) -> int:
    avg_chars_per_token = 3.5  # Based on LLaMA's ~3.5–4 char/token average
    return max(1, int(len(text) / avg_chars_per_token))

def should_append_observation(tool_name: str) -> bool:
    """Only append observations for tools that return useful static data."""
    return tool_name in {"memory", "search_web", "read_note", "calc"}

def safe_regex_search(pattern, text):
    try:
        return re.search(pattern, text)
    except re.error as e:
        print(f"❌ Regex error: {e} (pattern: {pattern})")
        return None

def clean_obs(obs):
    if isinstance(obs, str) and len(obs) > 200:
        return obs[:200] + "..."
    if isinstance(obs, dict):
        return {k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v for k, v in obs.items()}
    return obs

def generate_summary_pass(scratch_log: list[str], max_tokens=100) -> str:
    prompt = f"""
These are Kohana’s internal scratchpad thoughts. Summarize them briefly in 3–5 sentences while preserving her reasoning:

{"".join(scratch_log)}

Summary:
"""
    return llm_generate(prompt, max_new_tokens=max_tokens).strip()

def generate_scratch_pass(base_prompt: str, scratch_history: list[str], max_tokens=20) -> str:
    prompt = base_prompt + "\n" + "\n".join(scratch_history)
    return llm_generate(prompt, max_new_tokens=max_tokens)

def serialize_scratch_log(log, window=5):
    """Serialize the last `window` entries from the structured log."""
    lines = []
    for entry in log[-window:]:
        if entry["role"] == "thought":
            lines.append(entry["text"])
        elif entry["role"] == "observation":
            lines.append(json.dumps({"observation": entry["data"]}))
    return "\n".join(lines)


def extract_json_objects(text: str) -> list[dict]:
    """
    Safely extract flat JSON objects from LLM output.
    Ignores recursive/nested braces. Won't break on Python 3.12+.
    """
    # Clean up obvious wrapper junk first
    cleaned = re.sub(r'[⟨⟩<>/]*JSON[⟨⟩<>/]*', '', text)

    # Find all likely flat JSON objects (from '{' to '}')
    # This will NOT match nested objects correctly, but is safe for most LLM output
    pattern = r'\{[^{}]*\}'  # Match any non-nested JSON block

    matches = re.findall(pattern, cleaned)
    objs = []
    for m in matches:
        try:
            objs.append(json.loads(m))
        except Exception as e:
            print(f"⚠️ Skipped malformed JSON: {e} -> {m}")
    return objs

#Not used, but can remove the 2 in extract_json_objects2 below to make it active. Make sure to deactivate the other active one if you do.  
def extract_json_objects2(text: str) -> list[dict]:
    import re
    import json
    cleaned = re.sub(r'[⟨⟩<>/]*JSON[⟨⟩<>/]*', '', text)
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        obj = json.loads(cleaned[start:end+1])
        return [obj]
    except Exception as e:
        print(f"⚠️ Could not parse JSON: {e}")
        return []


def clean_raw_llm_output(raw: str, base_prompt: str) -> str:
    raw = raw.replace(base_prompt, "").replace("```json", "").replace("```", "").strip()
    raw = re.sub(r'[⟨⟩<>/]*JSON[⟨⟩<>/]*', '', raw)
    raw = re.sub(r'^REMINDER:.*$', '', raw, flags=re.MULTILINE)
    if "<|assistant|>" in raw:
        raw = raw.split("<|assistant|>")[-1].strip()
    if "⟨JSON⟩" in raw:
        raw = raw[raw.find("⟨JSON⟩"):]
    return raw

def build_prompt(base_prompt: str, scratch_log: list, reminder: bool = True) -> str:
    lines = []
    for entry in scratch_log[-5:]:
        if entry["role"] == "thought":
            lines.append(entry["text"])
        elif entry["role"] == "observation":
            lines.append(json.dumps({"observation": entry["data"]}))
    return (
        base_prompt + "\n" + "\n".join(lines) +
        ("\nREMINDER: Reply with exactly one ⟨JSON⟩…⟨/JSON⟩ line only." if reminder else "")
    )

def set_pending(conversation_state: dict, name: str, arg: str = "", guidance: str = ""):
    if conversation_state is not None:
        conversation_state.update({
            "pending_tool": name,
            "arg_so_far": arg,
            "arg_guidance": guidance
        })

def thought_log_only(scratch_log):
    return [e["text"] for e in scratch_log if e["role"] == "thought"]

def handle_tool_output(tool_name: str, raw_output) -> dict:
    """
    Safely process and type-check all tool outputs before passing to the scratch log, LLM, or UI.
    Returns a dict: { "type": "text"|"json"|"error"|"list"|"html", "content": ... }
    """
    # 1. Known tools that always return plain text
    if tool_name in {"calc", "note", "set_reminder", "set_timer", "read_note"}:
        content = str(raw_output)
        # Sanitize for braces/newlines just in case
        content = content.replace("{", "(").replace("}", ")").replace("\n", " ").strip()
        return {"type": "text", "content": content[:400] + "..." if len(content) > 400 else content}

    # 2. Tools that may return JSON or list (memory search, web search, etc)
    if tool_name in {"memory", "search_web"}:
        try:
            parsed = json.loads(raw_output)
            return {"type": "json", "content": parsed}
        except Exception:
            text = str(raw_output).replace("{", "(").replace("}", ")")
            return {"type": "text", "content": text[:400] + "..." if len(text) > 400 else text}

    # 3. For HTML in output (avoid ever passing raw to LLM)
    if "<html" in str(raw_output).lower():
        soup = BeautifulSoup(str(raw_output), "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return {"type": "text", "content": text[:400] + "..." if len(text) > 400 else text}

    # 4. Fallback
    text = str(raw_output)
    if len(text) > 400:
        text = text[:400] + "..."
    print(f"⚠️ [TOOL OUTPUT WARNING] Unexpected output type for {tool_name}: {type(raw_output)} - {raw_output}")
    return {"type": "text", "content": text}


def scratch_loop(user_msg: str, meta_json: str, conversation_state=None):
    """
    Kohana's next-gen scratch loop: meta-reflective, error-tolerant, extensible.
    Now with universal tool output sanitization.
    """
    import copy

    scratch_log = []
    tool_usage = Counter()
    pass_count = 0
    MAX_PASSES = 4
    MAX_ERRORS = 3
    MAX_TOOL_CALLS = 3
    consecutive_errors = 0
    last_action_type = None
    answer = None
    MAX_TOKENS_PER_PASS = 80  # Adjust as needed for your model

    # -- Resume any pending tool interaction (multi-turn tools) --
    if conversation_state and "pending_tool" in conversation_state:
        name = conversation_state.pop("pending_tool", "")
        arg_so_far = conversation_state.pop("arg_so_far", "")
        full_arg = f"{arg_so_far}|{user_msg.strip()}" if arg_so_far else user_msg.strip()
        result = run_tool(name, full_arg, conversation_state)
        safe_result = handle_tool_output(name, result)
        scratch_log.append({
            "role": "observation", 
            "data": f"[RESUME TOOL] {name} with input: {user_msg} → {safe_result['content']}"
        })
        return safe_result["content"], scratch_log

    base_prompt = SCRATCH_TEMPLATE.format(meta_json=meta_json, user_msg=user_msg)

    while pass_count < MAX_PASSES:
        pass_count += 1
        log_context = serialize_scratch_log(scratch_log, window=4)
        prompt = (
            base_prompt
            + "\n[Scratch Context]\n" + log_context
            + f"\n[Pass #{pass_count}]"
            + "\nThink aloud about the best next step. If a tool is needed, specify which and why. "
            + "If ready to answer, end with 'final:true'. Only one valid JSON object!"
        )

        llm_output = generate_scratch_pass(prompt, thought_log_only(scratch_log), max_tokens=MAX_TOKENS_PER_PASS)
        objs = extract_json_objects(clean_raw_llm_output(llm_output, base_prompt))
        if not objs:
            scratch_log.append({"role": "error", "text": f"⚠️ LLM produced invalid or empty output: {llm_output}"})
            consecutive_errors += 1
            if consecutive_errors >= MAX_ERRORS:
                scratch_log.append({"role": "error", "text": "FATAL: Too many parse errors."})
                break
            continue

        obj = objs[-1]
        if obj.get("final", False):
            answer = obj.get("answer") or obj.get("response") or obj.get("text")
            scratch_log.append({"role": "thought", "text": f"[FINAL ANSWER] {answer}"})
            break

        # ---- Tool/action/thought handling ----
        if "tool" in obj:
            tool = obj["tool"]
            arg = obj.get("arg", "")
            if tool_usage[tool] >= MAX_TOOL_CALLS:
                scratch_log.append({"role": "thought", "text": f"[TOOL THROTTLE] Max calls for '{tool}'."})
                continue
            tool_usage[tool] += 1
            tool_result = run_tool(tool, arg, conversation_state)
            safe_result = handle_tool_output(tool, tool_result)
            # Only append observation if the tool result matters for reasoning
            if should_append_observation(tool,):
                obs = clean_obs(safe_result["content"])
                scratch_log.append({"role": "observation", "data": {tool: obs}})
            else:
                scratch_log.append({"role": "thought", "text": f"[TOOL] {tool} run (result hidden for brevity)"})
            last_action_type = "tool"
            continue

        if "thought" in obj:
            scratch_log.append({"role": "thought", "text": obj["thought"]})
            last_action_type = "thought"
            continue

        # Catch any parsing/response oddities
        scratch_log.append({"role": "error", "text": f"[UNEXPECTED LLM OUTPUT] {obj}"})
        consecutive_errors += 1
        if consecutive_errors >= MAX_ERRORS:
            scratch_log.append({"role": "error", "text": "FATAL: Too many odd outputs."})
            break

        if pass_count > 2 and last_action_type == "thought" and scratch_log[-1]["text"] == scratch_log[-2]["text"]:
            scratch_log.append({"role": "thought", "text": "[META] Stalled, forcing LLM to choose a different action next."})
            base_prompt += "\n[FORCE ACTION] Choose a new plan or tool, do not repeat last thought."

    if not answer:
        answer = "[NO FINAL ANSWER – possible error or unfinished reasoning.]"

    return answer, scratch_log



#================================================================================================================#
_SAFE_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "pow": pow}

def _safe_eval(expr: str) -> str:
    import ast, operator, math

    allowed_bin = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
    }

    def _eval(node, depth=0):
        if depth > 20:
            raise ValueError("expression too deep")

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin:
            return allowed_bin[type(node.op)](
                _eval(node.left, depth+1), _eval(node.right, depth+1)
            )

        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand, depth+1)
            return val if isinstance(node.op, ast.UAdd) else -val

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _SAFE_FUNCS.get(node.func.id)
            if fn:
                return fn(*[_eval(arg, depth+1) for arg in node.args])

        raise ValueError("unsafe operation")

    try:
        tree = ast.parse(expr, mode="eval")
        return str(_eval(tree.body))
    except (ValueError, ZeroDivisionError) as e:
        return f"error: {e}"

def is_valid_url(url: str) -> bool:
    return re.match(r'^https?://\S+\.\S+', url) is not None

from bs4 import BeautifulSoup
# very tiny helpers so run_tool doesn’t throw
def web_search(query: str) -> str:
    try:
        url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(res.text, "html.parser")
        snippet = soup.find("div", class_="BNeawe")
        if snippet:
            text = snippet.text.strip()
            if len(text) > 400:
                text = text[:400] + "..."
            # Remove any accidental braces or newlines
            text = text.replace("{", "(").replace("}", ")").replace("\n", " ")
            return text
        return "No relevant result found."
    except Exception as e:
        return f"Error during web search: {e}"


def memory_search(query: str) -> str:
    results = memory_module.retrieve_relevant_memories(query, top_k=2)
    return results[0]["content"] if results else "no memory found"

def note_tool(arg):
    """Expected format: 'title|content'"""
    print("Entered Note Tool")
    try:
        title, content = arg.split("|", 1)
        save_note(title.strip(), content.strip())
        print(f"Note saved: {title.strip()}")
        return f"Note saved as '{title.strip()}'"
    except Exception as e:
        return f"Error saving note: {e}"
    
def reminder_tool(arg):
    """Expected format: '10 minutes|Check the oven'"""
    try:
        time_str, message = arg.split("|", 1)
        now = datetime.datetime.now()
        if "second" in time_str:
            num = int(time_str.split("second")[0].split()[-1])
            remind_time = now + datetime.timedelta(seconds=num)
        elif "minute" in time_str:
            num = int(time_str.split("minute")[0].split()[-1])
            remind_time = now + datetime.timedelta(minutes=num)
        elif "hour" in time_str:
            num = int(time_str.split("hour")[0].split()[-1])
            remind_time = now + datetime.timedelta(hours=num)
        else:
            remind_time = dateutil.parser.parse(time_str, fuzzy=True)
        schedule_reminder(remind_time, message.strip())
        return f"Reminder set for {remind_time.strftime('%H:%M:%S')} – \"{message.strip()}\""
    except Exception as e:
        return f"Couldn’t understand the reminder input: {e}"

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


def send_email_with_attachment(to_email, subject, message, attachment_path=None, from_email="" #personal email here, password= "" #personal code here):
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

    # Quick path: full message in one line via scratch tool
    if "|" in prompt and prompt.count("|") >= 2:
        try:
            to, subject, content = prompt.split("|", 2)
            send_email_with_attachment(to.strip(), subject.strip(), content.strip())
            return f"Quick email sent to {to.strip()}!"
        except Exception as e:
            return f"Failed to send quick email: {e}"

    # Multi-step fallback flow
    if not step:
        conversation_state["email_step"] = "awaiting_recipient"
        conversation_state["email_data"] = {}
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
        print(f"✅ Email sent to {email_data['to']} with subject '{email_data['subject']}'")
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

def handle_finance_interaction(prompt, conversation_state):
    # Simplified NLP — replace with proper extraction if you want to improve this
    if "summary" in prompt.lower():
        summary = get_summary_by_category()
        return "Here’s your financial summary:\n" + "\n".join([f"{cat}: ${amt:.2f}" for cat, amt in summary])
    
    if "show" in prompt.lower() or "transactions" in prompt.lower():
        txns = view_transactions()
        return "Here are your recent transactions:\n" + "\n".join([
            f"{d} | {c} | {desc} | ${a:.2f}" for _, d, c, desc, a in txns
        ])

    # Default to trying to add a transaction
    try:
        parts = prompt.split(" ")
        amount = None
        for word in parts:
            if "$" in word or word.replace(".", "").isdigit():
                amount = float(word.replace("$", ""))
                break

        if not amount:
            return "How much was the transaction?"

        category = "Uncategorized"
        description = " ".join([w for w in parts if not "$" in w and not w.replace(".", "").isdigit()])

        add_transaction(category=category, description=description, amount=amount)
        return f"Got it. Logged: {description} for ${amount:.2f} in {category}."

    except Exception as e:
        return f"⚠️ Error logging transaction: {str(e)}"
