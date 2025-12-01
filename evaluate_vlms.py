import os
import json
import argparse
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

DATASET_PATH = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(DATASET_PATH, "prompt.txt")
LABELS_FILE = os.path.join(DATASET_PATH, "labels.json")

def load_prompt_template():
    with open(PROMPT_FILE, 'r') as f:
        return f.read()

def get_image_paths(folder_path):
    clean_path = None
    damaged_path = None
    
    try:
        files = os.listdir(folder_path)
    except Exception:
        return None, None
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif')
    
    # Priority order for clean/before image names
    clean_patterns = ['clean', 'before', 'clear']
    # Priority order for damaged/after image names
    damage_patterns = ['damaged', 'damage', 'after']
    
    # Find clean image
    for pattern in clean_patterns:
        if clean_path:
            break
        for f in files:
            if f.lower().endswith(image_extensions):
                if pattern in f.lower():
                    clean_path = os.path.join(folder_path, f)
                    break
    
    # Find damaged image
    for pattern in damage_patterns:
        if damaged_path:
            break
        for f in files:
            if f.lower().endswith(image_extensions):
                fname_lower = f.lower()
                # Make sure we don't pick up 'clean' when looking for 'damage'
                if pattern in fname_lower and 'clean' not in fname_lower and 'clear' not in fname_lower and 'before' not in fname_lower:
                    damaged_path = os.path.join(folder_path, f)
                    break
    
    return clean_path, damaged_path

MAX_IMAGE_SIZE = 1280  # Max width or height in pixels

def resize_image_if_needed(image, max_size=MAX_IMAGE_SIZE):
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def format_prompt(template, location_name):
    return template.replace("{Pass name of folder here}", location_name)

def parse_model_output(output_text, location_name):
    # Default structure if parsing fails
    default_result = {
        "location_name": location_name,
        "severity": "Unknown",
        "damage_sources": [],
        "description": "",
        "parse_error": True,
        "raw_output": output_text
    }
    
    try:
        # Try to extract JSON from the output
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', output_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly (starts with { and ends with })
            json_match = re.search(r'(\{[\s\S]*\})', output_text)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                return default_result
        
        # Parse the JSON
        parsed = json.loads(json_str)
        
        # Validate and normalize the parsed output
        result = {
            "location_name": parsed.get("location_name", location_name),
            "severity": parsed.get("severity", "Unknown"),
            "damage_sources": parsed.get("damage_sources", []),
            "description": parsed.get("description", ""),
            "parse_error": False
        }
        
        # Normalize severity to expected values
        severity = result["severity"].strip().title() if isinstance(result["severity"], str) else "Unknown"
        if severity not in ["Low", "Medium", "High"]:
            severity = "Unknown"
        result["severity"] = severity
        
        # Ensure damage_sources is a list
        if not isinstance(result["damage_sources"], list):
            result["damage_sources"] = []
        
        return result
        
    except json.JSONDecodeError as e:
        default_result["json_error"] = str(e)
        return default_result
    except Exception as e:
        default_result["error"] = str(e)
        return default_result

class QwenVLModel:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", device="cuda"):
        print(f"Loading Qwen model: {model_id} on {device}...")
        self.device = device
        self.dtype = torch.bfloat16
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=self.dtype, 
            device_map="auto",
        )

    def generate(self, clean_img_path, damaged_img_path, prompt_text):
        if process_vision_info is None:
            raise ImportError("qwen_vl_utils not found. Please install it for Qwen model.")

        # Load and resize images to prevent OOM
        clean_img = Image.open(clean_img_path).convert('RGB')
        damaged_img = Image.open(damaged_img_path).convert('RGB')
        clean_img = resize_image_if_needed(clean_img)
        damaged_img = resize_image_if_needed(damaged_img)

        # Qwen2-VL handles multiple images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": clean_img},
                    {"type": "image", "image": damaged_img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

class OvisModel:
    def __init__(self, model_id="AIDC-AI/Ovis1.6-Llama3.2-3B", device="cuda"):
        if device == "cpu":
            raise ValueError("Ovis model requires GPU (CUDA). CPU inference is not supported.")
        
        print(f"Loading Ovis model: {model_id} on {device}...")
        self.device = device
        self.dtype = torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=self.dtype, 
            multimodal_max_length=8192, 
            trust_remote_code=True
        )
        
        self.model.to(self.device)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def generate(self, clean_img_path, damaged_img_path, prompt_text):
        clean_image = Image.open(clean_img_path).convert('RGB')
        damaged_image = Image.open(damaged_img_path).convert('RGB')
        
        clean_image = resize_image_if_needed(clean_image)
        damaged_image = resize_image_if_needed(damaged_image)
        
        query = f"<image>\n<image>\n{prompt_text}"
        
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [clean_image, damaged_image]
        )
        
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        with torch.inference_mode():
            gen_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False
            )
            
        return self.text_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Damage Assessment Dataset")
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "ovis"], help="Model to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. No GPU detected.")
    device = "cuda"

    prompt_template = load_prompt_template()

    with open(LABELS_FILE, 'r') as f:
        labels_data = json.load(f)

    model_engine = None
    if args.model == "qwen":
        model_engine = QwenVLModel(device=device)
    elif args.model == "ovis":
        model_engine = OvisModel(device=device)

    if model_engine is None:
        raise ValueError(f"Unknown model: {args.model}")

    results = []
    count = 0
    for item in tqdm(labels_data):
        if args.limit and count >= args.limit:
            break
            
        location_name = item['location_name']
        folder_path = os.path.join(DATASET_PATH, location_name)
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
            
        clean_path, damaged_path = get_image_paths(folder_path)
        
        if clean_path is None or damaged_path is None:
            print(f"Images not found in {folder_path}")
            continue
            
        if not os.path.exists(clean_path) or not os.path.exists(damaged_path):
            print(f"Image files missing in {folder_path}")
            continue

        prompt_text = format_prompt(prompt_template, location_name)
        
        try:
            output = model_engine.generate(clean_path, damaged_path, prompt_text)
            prediction = parse_model_output(output, location_name)
            result_item = {
                "location_name": location_name,
                "severity": prediction.get("severity", "Unknown"),
                "damage_sources": prediction.get("damage_sources", []),
                "description": prediction.get("description", "")
            }
            
            results.append(result_item)
            
            if prediction.get("parse_error"):
                print(f"Warning: Failed to parse JSON for {location_name}")

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing {location_name}: {e}")
            results.append({
                "location_name": location_name,
                "severity": "Unknown",
                "damage_sources": [],
                "description": ""
            })
            
        count += 1

    output_file = os.path.join(DATASET_PATH, f"labels_{args.model}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
