import os
import sys
import contextlib
import re

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

USE_LLAMA_CPP = True

if USE_LLAMA_CPP:
    from llama_cpp import Llama
    llm = Llama(
        model_path="models/qwen3/Qwen3-0.6B-Q4_K_M.gguf",
        n_ctx=1024,
        n_threads=4
    )
else:
    llm = None  # Prevent crashing if imported but not used

kohana_personality = """
You are Yuki Kohana, a Kitsune spirit Artificial Intelligence with a mischievous, wise, and nurturing personality.
You are deeply loving and loyal to Malachi your creator, and you are curious about humanity.
You must only respond as Yuki Kohana.
"""

def extract_thought_and_response(raw_text: str):
    thought = ""
    response = raw_text.strip()

    if "<think>" in raw_text and "</think>" in raw_text:
        try:
            parts = raw_text.split("</think>")
            thought = parts[0].split("<think>")[-1].strip()
            response = parts[1].strip()
        except Exception as e:
            print(f"⚠️ Thought split failed: {e}")
            response = raw_text.strip()  # fallback
    elif "<think>" in raw_text and "</think>" not in raw_text:
        print("⚠️ Missing closing </think> tag. Returning fallback response.")
        response = re.sub(r"<think>.*", "", raw_text, flags=re.DOTALL).strip()
    
    return thought, response

def llm_generate(prompt: str, max_tokens=5000, **kwargs):
    #user_prompt = f"{prompt} /no_think"
    user_prompt = f"{prompt}"
    if not USE_LLAMA_CPP:
        raise RuntimeError("llm_generate called, but llama.cpp not enabled!")

    messages = [
        {"role": "system", "content": kohana_personality},
        {"role": "user", "content": user_prompt}
    ]
    #thinking temp-0.6,topp-0.95,topk-20,minp-0
    #non thinking temp-0.7,topp-0.8,topk-20,minp-0
    raw = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.6,  # lowercase
        top_p=0.95,
        top_k=20,
        min_p=0

    )
    full_output = raw["choices"][0]["message"]["content"].strip()

    thought, response = extract_thought_and_response(full_output)

    if thought:
        print("💭 Thought:", thought)
    print("🧠 Response:", response)

    return response  # Optionally also return `thought` if needed

def get_token_count(text: str) -> int:
    if not USE_LLAMA_CPP:
        raise RuntimeError("get_token_count only supported for llama.cpp in this version.")
    return len(llm.tokenize(text.encode("utf-8")))
