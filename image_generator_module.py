# image_generator_module.py  ✨refined
from __future__ import annotations
import json, datetime, torch, cv2, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline, ControlNetModel
)
import logging, diffusers
logging.getLogger("diffusers").setLevel(logging.ERROR)


# ---------- presets -----------------------------------------------------------
PRESETS: dict[str, dict[str,str]] = {
    "default": {
        "medium":"digital painting",
        "style":"hyperrealistic, highly detailed",
        "website":"artstation",
        "resolution":"4k, ultra detailed",
        "extras":"",
        "color":"vibrant color palette",
        "lighting":"soft cinematic lighting"
    },
    "anime_painterly": {
        "medium":"anime illustration",
        "style":"painterly, cel-shading",
        "website":"deviantart",
        "resolution":"HD line-art + watercolor wash",
        "extras":"",
        "color":"soft pastels",
        "lighting":"backlight with soft glow"
    },
    "hyperreal_portrait":{
        "medium":"digital concept art",
        "style":"hyperrealistic, smooth shading",
        "website":"artstation",
        "resolution":"4k close-up",
        "extras":"",
        "color":"natural tones",
        "lighting":"studio lighting"
    },
    "tech_sketch":{
        "medium":"technical illustration",
        "style":"blueprint, wireframe",
        "website":"behance",
        "resolution":"1024×1024 diagram",
        "extras":"labelled components, minimal",
        "color":"grayscale",
        "lighting":"flat"
    }
}
NEG_PROMPT = (
    "blurry, deformed, bad anatomy, extra limbs, watermark, text, lowres, nsfw"
)

# ---------- light-weight cache -----------------------------------------------
_CACHED: dict[str, object] = {}
def _get_pipe(kind: str, **kw):
    """kind = sd, img2img, controlnet-canny/scribble/depth"""
    if kind in _CACHED:
        return _CACHED[kind]
    base = "./Local_Version2.0/stable-diffusion-v1-5"

    # Always use float16 to minimize CPU RAM / heat
    dtype = torch.float16

    if kind == "sd":
        pipe = StableDiffusionPipeline.from_pretrained(
            base, torch_dtype=dtype, safety_checker=None,
            feature_extractor=None, use_safetensors=True
        )
    elif kind == "img2img":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base, torch_dtype=dtype, safety_checker=None,
            feature_extractor=None, use_safetensors=True
        )
    elif kind.startswith("control-"):
        model = kind.split("-", 1)[1]
        cn = ControlNetModel.from_pretrained(
            f"./controlnet/{model}", torch_dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base, controlnet=cn, torch_dtype=dtype, safety_checker=None,
            feature_extractor=None, use_safetensors=True
        )
    else:
        raise ValueError(f"Unknown pipe {kind}")

    # Always force to CPU for safety
    pipe = pipe.to("cpu")

    print("✅ Pipeline loaded in CPU safe mode.")
    _CACHED[kind] = pipe
    return pipe

# ---------- prompt helpers ----------------------------------------------------
def build_prompt(subject:str, preset:str="default", **overrides) -> str:
    pf = {**PRESETS["default"], **PRESETS.get(preset, {}), **overrides}
    return ", ".join([subject, pf["medium"], pf["style"], pf["website"],
                      pf["resolution"], pf["extras"], pf["color"], pf["lighting"]])

def build_refine_prompt(base:str, preset:str="default",
                        extras:str="same pose, same colors") -> str:
    pf = {**PRESETS["default"], **PRESETS.get(preset, {})}
    return ", ".join([base, pf["style"], extras, pf["resolution"],
                      "preserve composition", pf["lighting"]])

# ---------- pipeline wrappers -------------------------------------------------
_TMP = Path("./image_tmp"); _TMP.mkdir(exist_ok=True)

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_initial(subject:str, preset:str="default")->Path:
    prompt = build_prompt(subject, preset)
    out = _TMP/f"init_{_timestamp()}.png"
    pipe = _get_pipe("sd")
    img = pipe(prompt,
            negative_prompt=NEG_PROMPT,
            guidance_scale=7.5,
            num_inference_steps=20,    # ← lowered from 30 to 20
            callback=lambda step, _, __: print(f" step {step}/20"),
            callback_steps=5).images[0]
    img.save(out); return out

def apply_canny(path:Path)->Path:
    img = cv2.imread(str(path)); img=cv2.resize(img,(512,512))
    edges = cv2.Canny(img,100,200); rgb = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    out = _TMP/f"canny_{_timestamp()}.png"; Image.fromarray(rgb).save(out); return out

def controlnet_gen(edge_path:Path, subject:str, cn_type:str="canny",
                   preset:str="default")->Path:
    prompt = build_prompt(subject, preset)
    pipe = _get_pipe(f"control-{cn_type}")
    edge = Image.open(edge_path).convert("RGB").resize((512,512))
    img = pipe(prompt, negative_prompt=NEG_PROMPT,
               image=edge, controlnet_conditioning_scale=0.8,
               guidance_scale=8.5, num_inference_steps=25).images[0]
    out=_TMP/f"cn_{_timestamp()}.png"; img.save(out); return out

def refine(path:Path, subject:str, preset:str="default", strength:float=0.28)->Path:
    prompt = build_refine_prompt(subject, preset)
    init=Image.open(path).convert("RGB").resize((512,512))
    pipe=_get_pipe("img2img")
    img=pipe(prompt, negative_prompt=NEG_PROMPT, image=init,
             strength=strength, guidance_scale=7.5,
             num_inference_steps=25).images[0]   # ← lowered from 40 to 25
    out=_TMP/f"final_{_timestamp()}.png"; img.save(out); return out

# ---------- high-level front door --------------------------------------------
def generate_image(
    subject:str,
    preset:str="default",
    controlnet:str|None=None,     # "canny"/"scribble"/"depth" or None
    refine_after:bool=True
)->str:
    """
    Returns the final PNG path (string) – ready to send to front-end.
    """
    init = generate_initial(subject, preset)

    if controlnet:
        canny = apply_canny(init) if controlnet=="canny" else init
        img   = controlnet_gen(canny, subject, controlnet, preset)
    else:
        img = init

    final = refine(img, subject, preset) if refine_after else img
    return str(final)

# ---------- optional: simple safety checker -----------------------------------
def looks_safe(path:str)->bool:
    # trivial detection: censor if too much skin-tone dominant
    im=np.array(Image.open(path).resize((128,128)))
    hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV); h,s,v=cv2.split(hsv)
    skin=((h<25)|(h>160))&(s>50)&(v>50)
    return skin.mean()<0.35   # >35 % skin -> flag

# ---------- tool entrypoint for Kohana ---------------------------------------
def tool_entry(arg_json:str)->str:
    """
    arg_json example:
      {"subject":"silver-haired fox girl","preset":"anime_painterly",
       "controlnet":"canny","refine":true}
    Returns: path/to/image.png  OR  error msg
    """
    try:
        cfg=json.loads(arg_json)
        path=generate_image(
            cfg["subject"],
            cfg.get("preset","default"),
            cfg.get("controlnet"),
            cfg.get("refine",True)
        )
        if not looks_safe(path):
            return "error: nsfw-risk detected"
        return {"type": "image", "text": f"/{path}"}
    except Exception as e:
        return f"error: {e}"
    
# ⬇️  add at bottom of the file
# -------------------------------------------------------------
def refine_existing(image_path: str, instructions: str = "") -> str:
    """
    Re-runs img2img on `image_path` with extra `instructions`
    and returns the new PNG path.
    """
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = Path(image_path).with_stem(
              Path(image_path).stem + f"_refined_{ts}").with_suffix(".png")

    # reuse the low-level refine() we wrote above
    refine(Path(image_path),
           subject=instructions or "same subject",
           preset="hyperreal_portrait",
           strength=0.25)

    return str(out)