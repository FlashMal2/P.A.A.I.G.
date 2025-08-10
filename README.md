PAAIG - Project Overview & Module Documentation
##Non-technical notes: I began the development of this architecture to try and create a locally hosted architecture that can compete with larger architectures. This is still early in development and I am consitently working on updates. I have a smaller version of the architecture running on a Raspberry Pi 5 16GB. The smaller version has less complex scratch logic and uses a quantized Qwen 3 500M. The Raspberry Pi version version has a toggle to turn on and off thinking mode for Qwen 3 and both perform, with non-thinking performing faster, but thinking certainly acting more intelligent than when toggled off. It also shows some signs of emergent behavior, which tells me that the architecture learns and not just the LLM holding the reigns on emergence. The biggest highlight of this version is the custom databse using FAISS and SQL that allows the architecture to remember conversations long term and pickup on thier users preferences over time. This also has a 'user toggle' that I included so that my kids could talk to the architecture over time, and is able to be modified to change preferences based on who is using the architecture (requires manual modification). The personality and name of the current personality can be manually overridden as well for various use-cases [current name Yuki Kohana, current user Malachi (me)] (i.e personality can be prompt engineered to be used for research, customer chat bots or other various use-cases). There is a journaling part as well that automatically creates journal entries over time, the interval can be modified as well as the reason for journaling [can be revised to use scratchpad for research or self reflection as needed]. 


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
