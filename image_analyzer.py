# image_analyzer.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import face_recognition
from difflib import SequenceMatcher
import os

# Load BLIP model (cached for performance)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load known face (assumes malachi_face.jpg is in root directory)
KNOWN_FACE_PATH = "malachi_face.jpg"
if os.path.exists(KNOWN_FACE_PATH):
    known_face_image = face_recognition.load_image_file(KNOWN_FACE_PATH)
    known_face_encoding = face_recognition.face_encodings(known_face_image)[0]
else:
    known_face_encoding = None


def generate_caption(image_path: str) -> str:
    """Generates a caption for an image using BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)


def check_for_malachi(image_path: str) -> bool:
    """Returns True if Malachi's face is found in the image."""
    if known_face_encoding is None:
        return False

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for encoding in unknown_encodings:
        if face_recognition.compare_faces([known_face_encoding], encoding)[0]:
            return True
    return False


def compare_prompt_to_caption(prompt: str, caption: str) -> float:
    """Compares prompt and caption and returns a similarity ratio (0.0 - 1.0)."""
    return SequenceMatcher(None, prompt.lower(), caption.lower()).ratio()


def analyze_image(image_path: str, prompt: str = None) -> dict:
    """Runs a full analysis on the image and returns a report."""
    caption = generate_caption(image_path)
    contains_malachi = check_for_malachi(image_path)
    similarity = compare_prompt_to_caption(prompt, caption) if prompt else None

    return {
        "caption": caption,
        "contains_malachi": contains_malachi,
        "similarity": similarity
    }


report = analyze_image("generated/kohana_forest.png", prompt="a fox spirit in a forest with cherry blossoms")
print(report)

if report["similarity"] and report["similarity"] < 0.5:
    print("Kohana: Hmm… that’s not quite right. Let me try again.")