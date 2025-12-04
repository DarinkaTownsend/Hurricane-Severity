

from transformers import AutoProcessor, AutoModel, AutoConfig
import torch
import os
import json
import argparse


# -------------------------------------------------------------------------
# Find clean and damaged image pairs
# -------------------------------------------------------------------------
def find_pair_images(folder_path):
    """Find clean* or damaged*/damage* image pairs."""
    clean = None
    damaged = None

    for f in os.listdir(folder_path):
        f_lower = f.lower()

        if f_lower.startswith("clean"):
            clean = os.path.join(folder_path, f)

        if f_lower.startswith("damage") or f_lower.startswith("damaged"):
            damaged = os.path.join(folder_path, f)

    return clean, damaged


# -------------------------------------------------------------------------
# Build damage analysis prompt
# -------------------------------------------------------------------------
def build_prompt(location_name):
    return f"""
You are an expert image analyst specializing in post-disaster damage assessment. Your task is to analyze a pair of images showing a location before and after a major hurricane. You will describe the damage, classify its severity and sources, and format the output as a single, valid JSON object.

Inputs:

    Location Name: {location_name}
    Before Images are labeled with a clean*.*
    After Images are labeled with a damaged*.*

Instructions

    Analyze and Describe: Carefully compare the 'before' and 'after' images. Write a detailed 2-3 sentence summary comparing the images. Explicitly mention specific objects (e.g., 'roof', 'trees', 'road', 'windows') and their condition (e.g., 'missing', 'submerged', 'broken', 'toppled'). Describe the spatial relationships where relevant (e.g., 'tree on house'). This will be the value for the "description" key.

    Classify Damage Severity: Based on the visual evidence, classify the overall damage severity. You must choose one category: Low, Medium, or High. This will be the value for the "severity" key.

    Identify Damage Sources: Based on the visual evidence, select all applicable sources from the list below. This will be for the "damage_sources" key.

Damage Severity Scale:
    Low, Medium, High

Potential Damage Sources:
    Rain, Storm Surge, High Wind, Fallen Trees, Fire, Heavy Rain, Lost Power

Output Format (strict)
IMPORTANT: Respond ONLY with a single valid JSON object. No text outside JSON.

{{
  "location_name": "{location_name}",
  "severity": "",
  "damage_sources": [],
  "description": ""
}}
"""


# -------------------------------------------------------------------------
# Run OmniVinci model on one folder
# -------------------------------------------------------------------------
def run_analysis_on_folder(model, processor, generation_config, folder_path):
    folder_name = os.path.basename(folder_path)

    before_path, after_path = find_pair_images(folder_path)

    if not before_path or not after_path:
        print(f"[SKIP] Missing clean/damaged images → {folder_name}")
        return None

    prompt = build_prompt(folder_name)

    # Images passed as paths (chat template loads them internally)
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": before_path},
            {"type": "image", "image": after_path},
            {"type": "text", "text": prompt}
        ]
    }]

    # Apply template -> full model input text
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    # Processor loads images from the chat template automatically
    inputs = processor(
        text=[text],
        return_tensors="pt"
    )

    # Move tensors to GPU
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    # Run generation
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        media=inputs["media"],               # OmniVinci-specific fields
        media_config=inputs["media_config"], # OmniVinci-specific fields
        generation_config=generation_config,
    )

    response = processor.tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()

    # Validate JSON
    try:
        parsed = json.loads(response)
        print(f"[OK] Parsed JSON → {folder_name}")
        return parsed
    except Exception:
        print(f"[ERROR] Invalid JSON → {folder_name}")
        return {
            "location_name": folder_name,
            "severity": "Unknown",
            "damage_sources": [],
            "description": response
        }


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch hurricane damage analysis (OmniVinci)")
    parser.add_argument("--model_path", type=str, default="./", help="Path to OmniVinci model")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="results_json")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Load model
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Generation config
    generation_config = model.default_generation_config
    generation_config.update(
        max_new_tokens=args.max_new_tokens,
        max_length=99999999
    )

    # List folders
    folders = sorted([
        os.path.join(args.dataset_path, f)
        for f in os.listdir(args.dataset_path)
        if os.path.isdir(os.path.join(args.dataset_path, f))
    ])

    all_results = []

    # Run model on each folder
    for folder in folders:
        result = run_analysis_on_folder(
            model, processor, generation_config,
            folder
        )
        if result is not None:
            all_results.append(result)

    # Save one big JSON file
    final_path = os.path.join(args.output_path, "all_results.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n[DONE] Saved all results → {final_path}")


# /home/fa202199/Downloads/Dataset_Final/Dataset_Final