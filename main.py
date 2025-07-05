from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
from pathlib import Path
from kohana_chat3 import generate_response, manage_kitsune_bucks, get_kitsune_bucks, run_cognitive_reflection_loop, send_notification_email, debug_email, llm_generate, load_scheduled_events, remove_scheduled_event, assistant_email, start_conversation, generate_response_casual
from fastapi.responses import FileResponse
from fastapi import Response
from starlette.responses import StreamingResponse
import subprocess
import os
import psutil
from typing import List
import threading
import json
import time
from fastapi import Request
import socket
from datetime import datetime
from base64 import b64decode
import traceback  
from fastapi import UploadFile, File
import whisper
import tempfile
from session_manager import update_session, load_trimmed_session
from pydantic import BaseModel

model = whisper.load_model("base")  # or "tiny" for faster performance on Pi

UPLOADS_DIR = Path("static/uploads")   #  <- move uploads inside ./static
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

last_interaction_time = time.time()
notification_email = "malachi.notifications.ai@gmail.com"

def error_messages(message):
    print(message)

def restore_scheduled_events():
    events = load_scheduled_events()
    now = datetime.now()

    for event in events:
        event_type = event["type"]
        trigger_time = datetime.fromisoformat(event["trigger_time"])
        message = event["message"]

        # Skip events already in the past
        if trigger_time < now:
            continue

        wait_seconds = (trigger_time - now).total_seconds()

        def event_thread(event_type=event_type, trigger_time=trigger_time, message=message):
            time.sleep(wait_seconds)
            try:
                subject = f"🦊✨Kohana {event_type.capitalize()} \n Reminder:\n\n{message}\n\nScheduled for: {trigger_time.strftime('%Y-%m-%d %H:%M:%S')}"
                error_messages(subject)
            except Exception as e:
                print(f"⚠️ Failed event email: {e}")

            # After it fires, remove it from the list
            remove_scheduled_event(event_type, trigger_time, message)

        t = threading.Thread(target=event_thread, daemon=True)
        t.start()

    print(f"✅ Restored {len(events)} scheduled events.")

def update_last_interaction():
    global last_interaction_time
    last_interaction_time = time.time()

def clean_lonely_message(raw_text):
    parts = raw_text.split("Kohana:")
    return parts[-1].strip() if len(parts) > 1 else raw_text.strip()

def get_current_mode():
    try:
        with open("current_mode.txt", "r") as f:
            mode = f.read().strip().lower()
            if mode not in ["casual", "work"]:
                print(f"❓ Unknown mode: {mode}")
                return "work"
            return mode
    except Exception as e:
        print(f"⚠️ Mode read failed, defaulting to work: {e}")
        return "work"



#def monitor_loneliness():
#    while True:
#        if time.time() - last_interaction_time > 900:  # 30 minutes no chat
#            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#            thoughts = start_conversation()
#            try:
#                loneliness_prompt = (
#                    f"### Instruction: \n"
#                    f"You are Yuki Kohana, Malachi's companion and Kitsune AI assistant.\n"
#                    f"You are reaching out to check on Malachi because you haven't talked to him in about an hour.\n"
#                    f"The current date and time is {now_str}.\n"
#                    f"Possible related memories: {thoughts}\n"
#                    f"Write your own message to begin the conversation.\n\n"
#                    f"### Response: \n"
#                    f"Kohana:"
#                )
#                kohana_lonely_message = llm_generate(loneliness_prompt, max_new_tokens=200)
#                kohana_lonely_message = clean_lonely_message(kohana_lonely_message)

                # 📝 Append to chat history
#                try:
#                    history_path = Path("chat_histories/malachi_history.json")
#                    if history_path.exists():
#                        with open(history_path, "r", encoding="utf-8") as f:
#                            history = json.load(f)
#                    else:
#                        history = []#
#
#                    history.append({
#                        "role": "kohana",
#                        "text": kohana_lonely_message,
#                        "type": "text"
#                    })

#                    with open(history_path, "w", encoding="utf-8") as f:
#                        json.dump(history, f, indent=2)
#                except Exception as e:
#                    error_messages(f"⚠️ Error saving lonely message to history: {e}")#

#                print("📩 Lonely alert sent to Malachi + added to chat!")
#                time.sleep(1800)  # Wait another 30 min after sending one
#            except Exception as e:
#                error_messages(f"⚠️ Error in loneliness monitor: {e}")
#        time.sleep(60)

def get_gpu_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            usage = result.stdout.strip()
            return f"{usage}%"
    except Exception as e:
        error_messages(f"⚠️ GPU fetch error: {e}")
    return "--"

def get_temperature():
    try:
        temps = psutil.sensors_temperatures()
        for label in ['coretemp', 'k10temp', 'cpu-thermal', 'acpitz']:
            if label in temps:
                for sensor in temps[label]:
                    if hasattr(sensor, 'current'):
                        return f"{sensor.current:.1f}°C"
    except Exception as e:
        error_messages(f"⚠️ Temperature fetch error: {e}")
    return "--"

def monitor_system_health():
    while True:
        try:
            temp_str = get_temperature()
            if temp_str != "--":
                temp_value = float(temp_str.replace("°C", ""))
                if temp_value >= 75:  # 🔥 Example: Over 75C considered overheating
                    error_messages(
                        subject="🔥 Kohana Overheat Warning! System temperature is critically high: {temp_value}°C"
                    )
                    time.sleep(600)  # Cooldown before checking again after alert
        except Exception as e:
            error_messages(f"⚠️ Error in health monitor: {e}")
        time.sleep(600)  # Check every 2 minutes


# Start loneliness monitor
#lonely_thread = threading.Thread(target=monitor_loneliness)
#lonely_thread.daemon = True
#lonely_thread.start()
health_thread = threading.Thread(target=monitor_system_health)
health_thread.daemon = True
health_thread.start()
# Store user histories
history_dir = Path("chat_histories")
history_dir.mkdir(exist_ok=True)

connected_ips = set()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

conversation_state = {}

#@app.on_event("startup")
#def run_cognitive_loop():
#    print("🌀 Spawning cognitive reflection thread from main...")
#    cognitive_thread = threading.Thread(
#        target=run_cognitive_reflection_loop,  # no need to pass anything if state is internal
#        daemon=True
#    )
#    cognitive_thread.start()



@app.on_event("startup")
def rehydrate_last_session():
    try:
        print("🔁 Loading last session memory...")
        trimmed = load_trimmed_session()

        if trimmed:
            restored_history = ""
            for msg in trimmed:
                restored_history += f"Malachi: {msg.get('user', '')}\n"
                restored_history += f"Kohana: {msg.get('kohana', '')}\n"
            conversation_state["chat_history"] = restored_history.strip()

            print("✅ Conversation state rehydrated.")
        else:
            print("ℹ️ No prior session found to restore.")
    except Exception as e:
        print(f"⚠️ Error restoring session: {e}")


@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": [],
        "kitsune_bucks": get_kitsune_bucks()
    })

@app.get("/os", response_class=HTMLResponse)
def os_mode(request: Request):
    return templates.TemplateResponse("os_mode.html", {"request": request})

@app.get("/list-events", response_class=JSONResponse)
async def list_scheduled_events():
    """
    Lists all upcoming scheduled events (reminders and timers).
    """
    events = load_scheduled_events()
    if not events:
        return {"events": []}

    upcoming_events = []
    now = datetime.now()

    for event in events:
        trigger_time = datetime.fromisoformat(event["trigger_time"])
        if trigger_time >= now:
            upcoming_events.append({
                "type": event["type"],
                "message": event["message"],
                "scheduled_for": trigger_time.strftime("%Y-%m-%d %H:%M:%S"),
                "in_seconds": int((trigger_time - now).total_seconds())
            })

    return {"events": upcoming_events}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        result = model.transcribe(temp_path)
        return {"text": result["text"]}
    except Exception as e:
        return {"error": str(e)}


@app.post("/save-history")
async def save_history(request: Request):
    try:
        data = await request.json()
        username = data.get("username", "Malachi")
        history = data.get("history", [])

        history_path = history_dir / f"{username}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return {"success": True}
    except Exception as e:
        error_messages(f"❌ Error saving history: {e}")
        return {"success": False, "error": str(e)}

@app.get("/get-history")
async def get_history(username: str = "Malachi"):
    try:
        history_path = history_dir / f"{username}_history.json"
        if not history_path.exists():
            return {"history": []}

        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        return {"history": history}
    except Exception as e:
        error_messages(f"❌ Error fetching history: {e}")
        return {"history": []}

@app.post("/restart_kohana")
def restart_kohana():
    subprocess.Popen(["sudo", "systemctl", "restart", "kohana.service"])
    return {"status": "Kohana service restarting..."}



@app.post("/chat")
async def chat_api(user_input: str = Form(...), username: str = Form(...), file: UploadFile = File(None)):
    print(f"📩 Incoming /chat request: user_input={user_input}, username={username}, file={file.filename if file else 'None'}")
    if "chat_history" not in conversation_state:
        print("🧠 No chat history found — initializing.")
        conversation_state["chat_history"] = ""
    
    attachment_path = None
    response_type = "text"
    response_text = ""

    if file:
        extension = file.filename.split('.')[-1].lower()
        allowed_types = {
            "image": ["jpg", "jpeg", "png", "gif", "webp"],
            "file": ["pdf", "txt", "md", "zip", "rar", "csv", "docx"]
        }

        # Save file
        file_location = UPLOADS_DIR / file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        attachment_path = f"/static/uploads/{file.filename}"

        # Classify the file
        if extension in allowed_types["image"]:
            conversation_state["last_user_image"] = attachment_path
            response_type = "image"
            response_text = f"[Image: {attachment_path}]"
        elif extension in allowed_types["file"]:
            response_type = "file"
            response_text = f"[File: {attachment_path}]"
        else:
            return JSONResponse(status_code=415, content={"error": "Unsupported file type."})

    if user_input:
        print(f"🗣️ User said: {user_input}")
        mode = get_current_mode()
        print(f"🎛️ Mode selected: {mode}")

        if mode == "casual":
            kohana_response = generate_response_casual(
                user_input,
                username,
                conversation_state=conversation_state
            )
        else:
            kohana_response = generate_response(
                user_input,
                conversation_state["chat_history"],
                username,
                conversation_state=conversation_state
            )

        if isinstance(kohana_response, dict):
            kohana_text = kohana_response["text"]
            kohana_type = kohana_response["type"]
        else:
            kohana_text = kohana_response
            kohana_type = "text"

        # Append to conversation history
        if not conversation_state.pop("skip_history", False):
            conversation_state["chat_history"] += (
                f"{username}: {user_input}\n"
                f"Kohana: {kohana_text}\n"
            )
        update_session(user_input, kohana_text)
        response_text = kohana_text
        response_type = kohana_type

    return JSONResponse(content={
        "response": response_text.strip(),
        "type": response_type
    })

class ModeUpdate(BaseModel):
    mode: str  # "casual" or "work"

@app.post("/update_mode")
def update_mode(data: ModeUpdate):
    with open("current_mode.txt", "w") as f:
        f.write(data.mode)
    return {"status": f"Mode set to {data.mode}"}

@app.post("/run-terminal")
def run_terminal(command: str = Form(...)):
    try:
        print(f"🧪 Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return {"stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reset-session")
async def reset_trimmed_session():
    try:
        trimmed_path = Path("last_session_trimmed.json")
        if trimmed_path.exists():
            trimmed_path.unlink()  # Delete the file
        conversation_state["chat_history"] = ""
        return {"success": True, "message": "Kohana's short-term memory has been reset."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/tool/files", response_class=HTMLResponse)
async def load_file_tool():
    html = """
    <div style="display:flex;flex-direction:column;gap:1rem;">
      <div><strong>File Browser</strong></div>
      <div id="file-path-display" style="font-size:0.9rem;">Current Path: <span id="current-path">.</span></div>
      <button onclick="navigateBack()" style="background:red;color:white;padding:6px;border:none;border-radius:5px;width:80px;">⬅️ Back</button>

      <div style="max-height:300px; overflow-y:auto; border:1px solid #333; padding:0.5rem; border-radius:6px;">
        <ul id="file-list" style="list-style:none;padding:0;margin:0;"></ul>
      </div>

      <hr style="border: 1px solid #444;" />

      <input id="file-path" placeholder="Enter file path..." style="background:#222;color:white;border:1px solid red;padding:8px;border-radius:6px;">
      <textarea id="file-editor" placeholder="File content..." style="background:#1e1e1e;color:#ccc;padding:10px;border-radius:6px;height:200px;font-family:monospace;"></textarea>
      <button onclick="saveFile()" style="background:green;color:white;padding:10px;border:none;border-radius:6px;">💾 Save</button>
    </div>
    <script>
    let currentPath = ".";
    
    console.log("🧪 File Tool script loaded!");

    function loadDirectory(path) {
        fetch('/list-dir?path=' + encodeURIComponent(path))
        .then(res => res.json())
        .then(data => {
            currentPath = data.path;
            document.getElementById('current-path').textContent = currentPath;

            const fileList = document.getElementById('file-list');
            fileList.innerHTML = "";

            data.files.forEach(file => {
            const li = document.createElement('li');
            li.textContent = file;
            li.style.padding = '6px 0';
            li.style.color = '#7cf';
            li.style.cursor = 'pointer';
            li.onclick = () => handleFileClick(`${data.path}/${file}`);
            fileList.appendChild(li);
            });
        });
    }

    function navigateBack() {
        const parts = currentPath.split('/');
        parts.pop();
        const backPath = parts.join('/') || '.';
        loadDirectory(backPath);
    }

    function handleFileClick(fullPath) {
        fetch('/read-file?path=' + encodeURIComponent(fullPath))
        .then(res => res.json())
        .then(data => {
            if (data.content !== undefined) {
            document.getElementById('file-editor').value = data.content;
            document.getElementById('file-path').value = fullPath;
            } else {
            loadDirectory(fullPath);
            }
        });
    }

    function saveFile() {
        const path = document.getElementById('file-path').value.trim();
        const content = document.getElementById('file-editor').value;

        if (!path) return alert('No path provided.');

        fetch('/save-file', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ path, content })
        })
        .then(res => res.json())
        .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            alert('Saved!');
            loadDirectory(currentPath);
        }
        });
    }

    // ✅ Trigger file load on script eval (dynamic injection compatible)
    loadDirectory("/home/malachi");
    </script>

    """
    return HTMLResponse(content=html)


@app.get("/list-dir")
def list_directory(path: str = "."):
    try:
        base_path = Path("/home/malachi")  # Change this if needed
        resolved_path = Path(path).resolve()

        # 🛡️ Safety: Prevent access above base_path
        if not str(resolved_path).startswith(str(base_path)):
            print(f"🚫 Blocked unsafe access to: {resolved_path}")
            resolved_path = base_path

        # 📁 If it's not a directory, fallback
        if not resolved_path.exists() or not resolved_path.is_dir():
            print(f"⚠️ Path invalid or not a directory: {resolved_path}")
            resolved_path = base_path

        print(f"📂 Listing directory: {resolved_path}")
        files = os.listdir(resolved_path)
        return {
            "path": str(resolved_path),
            "files": files
        }
    except Exception as e:
        error_messages(f"❌ Error listing {path}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


    
@app.post("/save-file")
def save_file(data: dict):
    try:
        with open(data["path"], "w", encoding="utf-8") as f:
            f.write(data["content"])
        return {"message": "File saved successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/read-file")
def read_file(path: str):
    try:
        if not os.path.isfile(path):
            return {"error": "Not a file"}
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        return {"error": str(e)}
    
# Live system stats
@app.get("/system-stats")
def get_system_stats():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "gpu": get_gpu_usage(),
        "temp": get_temperature()
    }

# Log connected IPs (simple memory-tracked for now)
connected_ips = set()

@app.get("/scan-network")
def scan_network():
    device_map = {}
    try:
        with open("device_nicknames.json", "r") as f:
            device_map = json.load(f)
    except FileNotFoundError:
        pass
    return {"devices": [{"ip": ip, "nickname": device_map.get(ip, "")} for ip in connected_ips]}

@app.post("/save-nickname")
def save_nickname(ip: str = Form(...), nickname: str = Form(...)):
    try:
        path = Path("device_nicknames.json")
        data = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        data[ip] = nickname
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return {"message": "Saved."}
    except Exception as e:
        return {"error": str(e)}
    
@app.middleware("http")
async def track_ips(request: Request, call_next):
    ip = request.client.host
    connected_ips.add(ip)
    return await call_next(request)

@app.get("/system-devices")
def get_devices():
    return {"devices": list(connected_ips)}

# File operations
@app.post("/rename-file")
def rename_file(old_path: str = Form(...), new_path: str = Form(...)):
    try:
        os.rename(old_path, new_path)
        return {"message": "Renamed successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/delete-file")
def delete_file(path: str = Form(...)):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return {"message": "Deleted successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/create-folder")
def create_folder(path: str = Form(...)):
    try:
        os.makedirs(path, exist_ok=True)
        return {"message": "Folder created"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/create-file")
def create_file(path: str = Form(...), content: str = Form("")):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"message": "File created"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
        return {"message": "File uploaded"}
    except Exception as e:
        return {"error": str(e)}

from fastapi.responses import HTMLResponse

@app.get("/ping")
def ping_target(target: str):
    try:
        result = subprocess.run(["ping", "-c", "4", target], capture_output=True, text=True)
        return {"output": result.stdout or result.stderr}
    except Exception as e:
        return {"output": f"Error: {str(e)}"}
    
@app.get("/my-ip")
def get_my_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "local_ip": local_ip}


@app.get("/tool/console", response_class=HTMLResponse)
async def load_console_tool():
    html = """
    <div style="display:flex; flex-direction:column; gap:1rem;">
      <div><strong>Console</strong> <span style="font-size: 0.8rem; color:#777;">(press Enter to run)</span></div>
      <div id="terminal-output" class="terminal-box"></div>
      <input id="terminal-command" placeholder="Type a command..." autofocus />
    </div>

    <style>
      .terminal-box {
        background-color: #000;
        color: #00ff7f;
        padding: 10px;
        font-family: monospace;
        font-size: 0.9rem;
        height: 250px;
        overflow-y: auto;
        border-radius: 6px;
        border: 1px solid #333;
        white-space: pre-wrap;
      }

      #terminal-command {
        background-color: #111;
        color: #0f0;
        border: 1px solid #333;
        padding: 8px;
        font-family: monospace;
        width: 100%;
        border-radius: 4px;
      }

      .terminal-line { margin-bottom: 4px; }
      .stderr { color: #ff5555; }
      .prompt { color: #888; }

      @keyframes blink {
        50% { opacity: 0; }
      }
    </style>
    """
    return HTMLResponse(content=html)

@app.get("/tool/network", response_class=HTMLResponse)
async def load_network_tool():
    html = """
    <div style="display:flex; flex-direction:column; gap:1rem;">
      <div><strong>Network Monitor</strong></div>
      <button onclick="scanNetwork()">🔍 Scan Network</button>
      <input id="ping-target" placeholder="Ping IP or Hostname..." />
      <button onclick="pingHost()">📶 Ping</button>
      <div id="network-output" class="terminal-box"></div>
      <button onclick="getLocalIP()">📡 My IP</button>
    </div>

    <script>
      function scanNetwork() {
        fetch('/scan-network')
          .then(res => res.json())
          .then(data => {
            document.getElementById('network-output').innerHTML =
              "<strong>Connected Devices:</strong><br>" +
              data.devices.map(device => `
                <div>
                  <strong>${device.nickname || device.ip}</strong>
                  <span style="color:#888; font-size:0.8rem;"> (${device.ip})</span>
                  <button onclick="renameDevice('${device.ip}')">✏️ Rename</button>
                </div>
              `).join('');
          });
      }

      function renameDevice(ip) {
        const nickname = prompt("Enter nickname for " + ip);
        if (!nickname) return;
        const formData = new FormData();
        formData.append("ip", ip);
        formData.append("nickname", nickname);

        fetch('/save-nickname', {
          method: 'POST',
          body: formData
        })
        .then(() => scanNetwork());
      }

      function pingHost() {
        const target = document.getElementById('ping-target').value.trim();
        if (!target) return;
        fetch('/ping?target=' + encodeURIComponent(target))
          .then(res => res.json())
          .then(data => {
            document.getElementById('network-output').innerHTML +=
              `<div><strong>Ping ${target}:</strong><br>${data.output}</div>`;
          });
      }

      function getLocalIP() {
        fetch('/my-ip')
            .then(res => res.json())
            .then(data => {
            document.getElementById('network-output').innerHTML +=
                `<div><strong>Your IP:</strong> ${data.local_ip} (${data.hostname})</div>`;
            });
        }
    </script>
    """
    return HTMLResponse(content=html)

@app.get("/tool/editor", response_class=HTMLResponse)
async def load_editor_tool():
    html = """
    <div style="display:flex; flex-direction:column; gap:1rem;">
      <div><strong>Code Editor</strong></div>
      <input id="edit-path" placeholder="Enter file path..." />
      <textarea id="edit-content" style="height:300px;background:#111;color:#0f0;font-family:monospace;padding:10px;"></textarea>
      <div style="display:flex; gap:1rem;">
        <button onclick="loadCode()">📂 Load</button>
        <button onclick="saveCode()">💾 Save</button>
      </div>
      <div id="editor-status"></div>
    </div>
    <select id="syntax-mode">
        <option value="plaintext">Plaintext</option>
        <option value="python">Python</option>
        <option value="javascript">JavaScript</option>
        <option value="html">HTML</option>
    </select>

    <script>
      function loadCode() {
        const path = document.getElementById('edit-path').value.trim();
        fetch('/read-file?path=' + encodeURIComponent(path))
          .then(res => res.json())
          .then(data => {
            if (data.content !== undefined) {
              document.getElementById('edit-content').value = data.content;
              document.getElementById('editor-status').textContent = 'File loaded.';
            } else {
              document.getElementById('editor-status').textContent = 'Error loading file.';
            }
          });
      }

      function saveCode() {
        const path = document.getElementById('edit-path').value.trim();
        const content = document.getElementById('edit-content').value;
        fetch('/save-file', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ path, content })
        })
        .then(res => res.json())
        .then(data => {
          document.getElementById('editor-status').textContent = data.message || data.error;
        });
      }
    </script>
    """
    return HTMLResponse(content=html)



@app.get("/tool/logs", response_class=HTMLResponse)
async def load_logs_tool(request: Request):
    return templates.TemplateResponse("logs.html", {"request": request})

@app.get("/static/audio/kohana_response.wav")
def serve_kohana_wav():
    return FileResponse("static/audio/kohana_response.wav", media_type="audio/wav")

@app.get("/audio/kohana_response.mp3")
async def serve_kohana_audio():
    path = "static/audio/kohana_response.mp3"
    if not os.path.exists(path):
        return Response(status_code=404, content="Not Found")

    return FileResponse(
        path,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Accept-Ranges": "none"  # 🧠 Disables partial content responses
        }
    )

# Kitsune Bucks
@app.get("/get_bucks")
def get_bucks():
    bucks = get_kitsune_bucks()
    print(f"👛 Kitsune Bucks Retrieved: {bucks}")
    return {"kitsune_bucks": bucks}

@app.post("/add_bucks")
def add_bucks():
    new_val = manage_kitsune_bucks("add", 10)
    return {"kitsune_bucks": new_val}

@app.post("/remove_bucks")
def remove_bucks():
    new_val = manage_kitsune_bucks("remove", 10)
    return {"kitsune_bucks": new_val}


@app.post("/add-user")
async def add_user(request: Request):
    try:
        data = await request.json()
        username = data.get("username", "").strip().lower()
        name = data.get("name", "").strip()
        relationship = data.get("relationship", "User").strip()
        tone = data.get("tone", "Neutral").strip()

        if not username or not name:
            return {"success": False, "message": "Username and Name are required."}

        profile_path = Path("user_profiles.json")
        if not profile_path.exists():
            return {"success": False, "message": "user_profiles.json not found."}

        with open(profile_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)

        if username in profiles:
            return {"success": False, "message": f"User '{username}' already exists."}

        profiles[username] = {
            "name": name,
            "relationship": relationship,
            "tone": tone,
            "role": "User",
            "pronouns": "they/them",
            "flags": {"default_user": False},
            "tags": []
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4)

        return {"success": True, "message": f"User '{name}' added successfully."}

    except Exception as e:
        return {"success": False, "message": f"Failed to add user: {str(e)}"}

@app.get("/tool/{tool_name}")
async def load_tool_view(tool_name: str):
    try:
        html_map = {
            "console": "<div><strong>Console:</strong><br><input id='terminal-command' placeholder='Enter command'><button onclick='runTerminal()'>Run</button><pre id='terminal-output'>...</pre></div>",
            "files": "<div><strong>Files:</strong><br>File browser UI placeholder</div>",
            "monitor": "<div><strong>Monitor:</strong><br>Live system stats will appear here.</div>",
            "editor": "<div><strong>Editor:</strong><br>Editor UI placeholder</div>",
            "logs": "<div><strong>Logs:</strong><br>System log viewer placeholder</div>",
            "network": "<div><strong>Network:</strong><br>IP and device scan results</div>",
        }
        return HTMLResponse(content=html_map.get(tool_name, "<div>Tool not found.</div>"))
    except Exception as e:
        return HTMLResponse(content=f"<div>Error loading tool: {str(e)}</div>")

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
