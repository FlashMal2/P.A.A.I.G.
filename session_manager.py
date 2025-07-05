import json
import os
from datetime import datetime

# Configurable constants
TRIMMED_LOG_FILE = "last_session_trimmed.json"
FULL_LOG_DIR = "session_logs"
TOKEN_LIMIT = 2000  # Approximate max token count for context on reload

# Ensure log directory exists
os.makedirs(FULL_LOG_DIR, exist_ok=True)

# Approximate token counting function (adjustable)
def count_tokens(text):
    return int(len(text.split()) * 0.75)

# Append a new message pair and reflection
def update_session(user_input, kohana_response, reflection=None):
    today = datetime.now().strftime("%Y-%m-%d")
    full_log_path = os.path.join(FULL_LOG_DIR, f"full_log_{today}.json")
    trimmed_log = []

    # Load existing full log if exists
    full_log = []
    if os.path.exists(full_log_path):
        with open(full_log_path, "r") as f:
            full_log = json.load(f)

    # New entry structure
    entry = {
        "user": user_input,
        "kohana": kohana_response,
    }
    if reflection:
        entry["reflection"] = reflection

    # Append to full log
    full_log.append(entry)
    with open(full_log_path, "w") as f:
        json.dump(full_log, f, indent=4)

    # Load existing trimmed log if exists
    if os.path.exists(TRIMMED_LOG_FILE):
        with open(TRIMMED_LOG_FILE, "r") as f:
            trimmed_log = json.load(f)

    # Append new entry to trimmed log
    trimmed_log.append(entry)

    # Trim to token limit
    while sum(count_tokens(json.dumps(m)) for m in trimmed_log) > TOKEN_LIMIT:
        trimmed_log.pop(0)

    with open(TRIMMED_LOG_FILE, "w") as f:
        json.dump(trimmed_log, f, indent=4)

# Load the trimmed session context on reboot
def load_trimmed_session():
    if os.path.exists(TRIMMED_LOG_FILE):
        with open(TRIMMED_LOG_FILE, "r") as f:
            return json.load(f)
    return []

# Example usage:
# update_session("How are you feeling?", "I'm excited to help you today!", "Reflection: Malachi seems hopeful.")
# trimmed = load_trimmed_session()
# print(trimmed)

