import os
import json
import argparse
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
)

# ================== PATHS ==================

BASE_DATASET_DIR = "/home/an768196/Downloads/Dataset_Final_Images/Dataset_Final"
PROMPT_TEMPLATE_PATH = os.path.join(BASE_DATASET_DIR, "prompt.txt")
GROUND_TRUTH_LABELS_PATH = os.path.join(BASE_DATASET_DIR, "labels.json")

IMAGE_MAX_DIM = 1280  # Max width or height in pixels


# ================== UTILS ==================

def read_prompt_template_file() -> str:
    """Load the prompt template from disk."""
    with open(PROMPT_TEMPLATE_PATH, "r") as prompt_f:
        return prompt_f.read()


def find_image_pair_in_folder(folder_dir: str):
    """Return (clean_image_path, damaged_image_path) for a given folder."""
    clean_image_path = None
    damaged_image_path = None

    try:
        folder_files = os.listdir(folder_dir)
    except Exception:
        return None, None

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")

    # Priority order for clean/before image names
    clean_name_hints = ["clean", "before", "clear"]
    # Priority order for damaged/after image names
    damaged_name_hints = ["damaged", "damage", "after"]

    # Find clean image
    for hint in clean_name_hints:
        if clean_image_path:
            break
        for fname in folder_files:
            if fname.lower().endswith(image_exts):
                if hint in fname.lower():
                    clean_image_path = os.path.join(folder_dir, fname)
                    break

    # Find damaged image
    for hint in damaged_name_hints:
        if damaged_image_path:
            break
        for fname in folder_files:
            if fname.lower().endswith(image_exts):
                lower_name = fname.lower()
                # Avoid picking "clean"/"before"/"clear" for damaged
                if (
                    hint in lower_name
                    and "clean" not in lower_name
                    and "clear" not in lower_name
                    and "before" not in lower_name
                ):
                    damaged_image_path = os.path.join(folder_dir, fname)
                    break

    return clean_image_path, damaged_image_path


def maybe_resize_image(input_image: Image.Image, max_side: int = IMAGE_MAX_DIM) -> Image.Image:
    """Resize image if either dimension exceeds max_side, preserving aspect ratio."""
    width, height = input_image.size
    if width <= max_side and height <= max_side:
        return input_image

    if width > height:
        new_width = max_side
        new_height = int(height * (max_side / width))
    else:
        new_height = max_side
        new_width = int(width * (max_side / height))

    return input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def build_location_prompt(prompt_template: str, location_name: str) -> str:
    """Insert the location/folder name into the prompt template."""
    return prompt_template.replace("{Pass name of folder here}", location_name)


def decode_and_normalize_model_output(raw_text: str, location_name: str):
    """
    Parse the model's raw text output into a structured dictionary.

    Expected JSON format:

    {
      "location_name": "...",
      "severity": "Low/Medium/High",
      "damage_sources": [...],
      "description": "..."
    }
    """
    fallback_result = {
        "location_name": location_name,
        "severity": "Unknown",
        "damage_sources": [],
        "description": "",
        "parse_error": True,
        "raw_output": raw_text,
    }

    try:
        # Try to extract JSON between ``` ``` first
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        if fenced_match:
            json_payload_str = fenced_match.group(1).strip()
        else:
            # Fallback: first {...} block
            brace_match = re.search(r"(\{[\s\S]*\})", raw_text)
            if brace_match:
                json_payload_str = brace_match.group(1).strip()
            else:
                return fallback_result

        parsed_output = json.loads(json_payload_str)

        parsed_result = {
            "location_name": parsed_output.get("location_name", location_name),
            "severity": parsed_output.get("severity", "Unknown"),
            "damage_sources": parsed_output.get("damage_sources", []),
            "description": parsed_output.get("description", ""),
            "parse_error": False,
        }

        # Normalize severity
        sev_value = (
            parsed_result["severity"].strip().title()
            if isinstance(parsed_result["severity"], str)
            else "Unknown"
        )
        if sev_value not in ["Low", "Medium", "High"]:
            sev_value = "Unknown"
        parsed_result["severity"] = sev_value

        if not isinstance(parsed_result["damage_sources"], list):
            parsed_result["damage_sources"] = []

        return parsed_result

    except json.JSONDecodeError as json_err:
        fallback_result["json_error"] = str(json_err)
        return fallback_result
    except Exception as generic_err:
        fallback_result["error"] = str(generic_err)
        return fallback_result


# ================== MODEL WRAPPERS ==================

class LlavaOneVisionWrapper:
    """
    Wrapper around: llava-hf/llava-onevision-qwen2-7b-ov-hf
    """

    def __init__(
        self,
        model_identifier: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        device_type: str = "cuda",
    ):
        print(f"Loading LLaVA-OneVision model: {model_identifier} on {device_type}...")
        self.device_type = device_type

        self.processor = AutoProcessor.from_pretrained(model_identifier)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_identifier,
            torch_dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
            device_map="auto" if device_type == "cuda" else None,
        )

    def run_inference(
        self,
        clean_image_path: str,
        damaged_image_path: str,
        user_prompt: str,
    ) -> str:
        """Generate raw text output given clean + damaged images and prompt."""
        # Load & resize
        clean_image = Image.open(clean_image_path).convert("RGB")
        damaged_image = Image.open(damaged_image_path).convert("RGB")

        clean_image = maybe_resize_image(clean_image)
        damaged_image = maybe_resize_image(damaged_image)

        # LLaVA-OneVision: text + list of images
        model_inputs = self.processor(
            text=user_prompt,
            images=[clean_image, damaged_image],
            return_tensors="pt",
        ).to(self.device_type)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        decoded_output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return decoded_output


class Phi3VisionWrapper:
    """
    Wrapper around: microsoft/Phi-3-vision-128k-instruct
    """

    def __init__(
        self,
        model_identifier: str = "microsoft/Phi-3-vision-128k-instruct",
        device_type: str = "cuda",
    ):
        print(f"Loading Phi-3-Vision model: {model_identifier} on {device_type}...")
        self.device_type = device_type

        # Disable FlashAttention2 via _attn_implementation="eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            torch_dtype=torch.bfloat16,  # or "auto"
            trust_remote_code=True,
            _attn_implementation="eager",
        ).to(self.device_type)

        self.processor = AutoProcessor.from_pretrained(
            model_identifier,
            trust_remote_code=True,
        )

    def run_inference(
        self,
        clean_image_path: str,
        damaged_image_path: str,
        user_prompt: str,
    ) -> str:
        """Generate raw text output using Phi-3-Vision."""
        # Load & resize
        clean_image = Image.open(clean_image_path).convert("RGB")
        damaged_image = Image.open(damaged_image_path).convert("RGB")

        clean_image = maybe_resize_image(clean_image)
        damaged_image = maybe_resize_image(damaged_image)

        # Build chat-style prompt for Phi-3
        chat_messages = [
            {
                "role": "user",
                "content": "<|image_1|>\n<|image_2|>\n" + user_prompt,
            }
        ]

        chat_prompt = self.processor.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Pass images in same order as <|image_1|>, <|image_2|>
        model_inputs = self.processor(
            chat_prompt,
            [clean_image, damaged_image],
            return_tensors="pt",
        ).to(self.device_type)

        # No temperature/top_p/top_k, disable cache to avoid DynamicCache bug
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=False,
            )

        # Remove the prompt tokens from output
        gen_only_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]

        decoded_response = self.processor.batch_decode(
            gen_only_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return decoded_response


# ================== MAIN LOOP ==================

def run_vlm_evaluation():
    arg_parser = argparse.ArgumentParser(
        description="Evaluate VLMs on Damage Assessment Dataset"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llava", "phi3"],
        help="Model to use: 'llava' or 'phi3'",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images for testing",
    )
    cli_args = arg_parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. No GPU detected.")
    compute_device = "cuda"

    prompt_template = read_prompt_template_file()

    # labels.json = ground truth list (for location_name ordering)
    with open(GROUND_TRUTH_LABELS_PATH, "r") as labels_f:
        ground_truth_entries = json.load(labels_f)

    # Select model wrapper
    if cli_args.model == "llava":
        vlm_runner = LlavaOneVisionWrapper(device_type=compute_device)
    elif cli_args.model == "phi3":
        vlm_runner = Phi3VisionWrapper(device_type=compute_device)
    else:
        raise ValueError(f"Unknown model: {cli_args.model}")

    predictions_list = []
    processed_count = 0

    for label_entry in tqdm(ground_truth_entries):
        if cli_args.limit and processed_count >= cli_args.limit:
            break

        location_name = label_entry["location_name"]
        location_folder = os.path.join(BASE_DATASET_DIR, location_name)

        if not os.path.exists(location_folder):
            print(f"Folder not found: {location_folder}")
            continue

        clean_img_path, damaged_img_path = find_image_pair_in_folder(location_folder)

        if clean_img_path is None or damaged_img_path is None:
            print(f"Images not found in {location_folder}")
            continue

        if not os.path.exists(clean_img_path) or not os.path.exists(damaged_img_path):
            print(f"Image files missing in {location_folder}")
            continue

        prompt_for_location = build_location_prompt(prompt_template, location_name)

        try:
            raw_model_output = vlm_runner.run_inference(
                clean_img_path, damaged_img_path, prompt_for_location
            )
            parsed_prediction = decode_and_normalize_model_output(
                raw_model_output, location_name
            )

            prediction_entry = {
                "location_name": location_name,
                "severity": parsed_prediction.get("severity", "Unknown"),
                "damage_sources": parsed_prediction.get("damage_sources", []),
                "description": parsed_prediction.get("description", ""),
            }

            predictions_list.append(prediction_entry)

            if parsed_prediction.get("parse_error"):
                print(f"Warning: Failed to parse JSON for {location_name}")

            torch.cuda.empty_cache()

        except Exception as run_err:
            print(f"Error processing {location_name}: {run_err}")
            predictions_list.append(
                {
                    "location_name": location_name,
                    "severity": "Unknown",
                    "damage_sources": [],
                    "description": "",
                }
            )

        processed_count += 1

    output_json_path = os.path.join(
        BASE_DATASET_DIR, f"labels_{cli_args.model}.json"
    )
    with open(output_json_path, "w") as out_f:
        json.dump(predictions_list, out_f, indent=2)

    print(f"\nProcessing complete. Results saved to {output_json_path}")


if __name__ == "__main__":
    run_vlm_evaluation()
