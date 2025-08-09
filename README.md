PAAIG - Project Overview & Module Documentation
PAAIG is a modular, locally-hosted AI assistant designed for conversational intelligence, situational awareness, creative tools, and adaptive memory systems. It integrates a large language model (LLM) with long-term memory, environmental awareness, and optional generative capabilities like Stable Diffusion.

Getting Started
Requirements:
    • Python 3.10+
    • FastAPI, Uvicorn
    • psutil, FAISS, SQLAlchemy
    • NVIDIA GPU (for Stable Diffusion & GPU stats)
Installation:
pip install -r requirements.txt
Run Kohana:
python main.py
Then open http://localhost:8000 in your browser.

Main Application (main.py)
The entry point for running Kohana. Initializes FastAPI routes, handles the web UI (chat.html, os_mode.html), and connects user requests to the AI logic.
Responsibilities:
    • Starts the web server.
    • Routes API requests to the LLM.
    • Loads chat and OS mode interfaces.

Core LLM & Compiler Logic (kohana_chat3.py)
Manages conversation flow, reasoning, and structured output.
Features:
    • Processes user input through the LLM.
    • Uses a compiler step for structured responses.
    • Integrates situational awareness, memory, and tool calls.
    • Supports reflection and emergent behavior.

Memory Management (memory_manager.py)
Hybrid FAISS + SQL database for semantic and structured recall.
Features:
    • Vector search for semantic memory.
    • SQL for structured, tagged memories.
    • Supports emotional tagging and decay logic.

Session Management (session_manager.py)
Stores and trims session history for token-efficient context reloads.
Features:
    • Daily full log rotation.
    • Trimmed logs for fast context reloading.
Example:
from session_manager import update_session, load_trimmed_session
update_session("Hello", "Hi there!", "Reflection: Engaged")
print(load_trimmed_session())

Situational Awareness (situational_awareness.py)
Gathers machine and environment details for contextual AI responses.
Captures:
    • Time of day
    • Hostname, IP
    • OS details
    • CPU/memory usage
    • GPU stats via nvidia-smi
    • Client type detection

Image Generation (image_generator_module.py)
Wrapper for Stable Diffusion + ControlNet with style presets.
Features:
    • Txt2Img & Img2Img
    • ControlNet support
    • Style presets & negative prompts
Example:
args = {"subject": "kitsune girl", "preset": "anime", "refine": True}
result = tool_entry(json.dumps(args))

Index
    1. Main Application
    2. Core LLM & Compiler Logic
    3. Memory Management
    4. Session Management
    5. Situational Awareness
    6. Image Generation
