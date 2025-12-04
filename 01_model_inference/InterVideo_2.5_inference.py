import os
import re
import json
import json5
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

#########################################
# MODEL LOADING
#########################################

model_path = "/home/fa202199/Downloads/vlms/InternVideo2_5_Chat_8B"
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)
model.eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

#########################################
# TRANSFORMS + PATCHING
#########################################

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def dynamic_preprocess_fixed_1patch(image, image_size=448):
    return [image.resize((image_size, image_size))]

def load_image(image, input_size=448):
    transform = build_transform(input_size)
    tiles = dynamic_preprocess_fixed_1patch(image)

    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, 1

#########################################
# JSON SAFETY UTILITIES
#########################################

def extract_json_block(text):
    """Extract first complete {...} using brace counting."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    return None

def clean_json(s):
    """Fix common VLM JSON mistakes."""
    # remove markdown fences
    s = s.replace("```json", "").replace("```", "").strip()

    # force double quotes around keys
    s = re.sub(r'(\w+)\s*:', r'"\1":', s)

    # replace single with double quotes
    # s = s.replace("", '"')

    # remove trailing commas inside {} or []
    s = re.sub(r',\s*([}\]])', r'\1', s)

    return s

#########################################
# PROMPT
#########################################

def build_hurricane_prompt(location_name: str) -> str:
    return f"""

    You are an expert image analyst specializing in post-disaster damage assessment. Your task is to analyze a pair of images showing a location before and after a major hurricane. You will describe the damage, classify its severity and sources, and format the output as a single, valid JSON object.

Inputs:
    Location Name: {location_name}
    Before Images are labeled with clean*.*
    After Images are labeled with damaged*.*

Instructions:

Analyze and Describe:
Compare the before and after images in 2–3 sentences. Mention specific objects (roof, trees, road, cars, windows), their conditions (missing, broken, flooded, toppled), and spatial relations (tree on roof, road underwater). This becomes the "description" field.

Classify Damage Severity:
Choose only one: Low, Medium, or High.

Identify Damage Sources:
Pick all that apply:
Rain, Storm Surge, High Wind, Fallen Trees, Fire, Heavy Rain, Lost Power.

Output Format:
Return ONLY a valid JSON object similar to something below:

{{
  "location_name": "{location_name}",
  "severity": "Low",
  "damage_sources": ["Storm Surge"],
  "description": "A detailed comparison."
}}

You must return ONLY valid JSON.

Rules:
- All keys MUST use double quotes.
- All values MUST use double quotes except lists.
- No trailing commas.
- No markdown formatting.
- The result MUST be valid JSON parseable by json.loads().
- Don't use any characters such as double quotes which makes the json invalid.
""".strip()

#########################################
# RUN MODEL
#########################################

generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=512,
    top_p=0.1,
    num_beams=1,
)

def run_pair(before_img, after_img, location_name):

    before_pix, _ = load_image(before_img)
    after_pix, _ = load_image(after_img)

    pixel_values = torch.cat(
        [before_pix, after_pix, before_pix, after_pix], dim=0
    )

    num_patches_list = [1, 1, 1, 1]
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(4)])
    prompt = video_prefix + build_hurricane_prompt(location_name)

    with torch.no_grad():
        output, _ = model.chat(
            tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )

    return output

def escape_inner_quotes(json_str):
    # Only fix the description field
    def repl(match):
        desc = match.group(1)
        desc_escaped = desc.replace('"', '\\"')
        return f'"description": "{desc_escaped}"'
    
    return re.sub(r'"description"\s*:\s*"(.*?)"', repl, json_str, flags=re.DOTALL)



#########################################
# MAIN LOOP
#########################################

DATASET_DIR = "/home/fa202199/Downloads/Dataset_Final/Dataset_Final"
# DATASET_DIR = "/home/fa202199/Downloads/xDB_train_images_labels_targets/train/sample_data_vlm"
all_results = []

def find_before_after(folder_path):
    before, after = None, None
    for fname in os.listdir(folder_path):
        lf = fname.lower()
        if "clean" in lf:
            before = os.path.join(folder_path, fname)
        if "damage" in lf or "damaged" in lf:
            after = os.path.join(folder_path, fname)
    return before, after

for folder_name in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    before_path, after_path = find_before_after(folder_path)
    if not before_path or not after_path:
        print(f"[!] Skipping {folder_name}: missing clean/damaged")
        continue

    print(f"Processing folder: {folder_name}")
    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")

    raw = run_pair(before_img, after_img, folder_name)

    json_raw = extract_json_block(raw)
    if not json_raw:
        print(f"[!] No JSON in VLM output for {folder_name}")
        continue

    cleaned = clean_json(json_raw)
    cleaned = escape_inner_quotes(cleaned)

    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        print(f"[!] JSON parse failed for {folder_name}: {e}")
        print("RAW JSON BLOCK:", cleaned)
        continue

    output = {
        "location_name": parsed.get("location_name", folder_name),
        "severity": parsed.get("severity", ""),
        "damage_sources": parsed.get("damage_sources", []),
        "description": parsed.get("description", ""),
    }

    all_results.append(output)

with open("my_pred.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("Saved results to my_pred.json")
