"""
Main entry point for VLM evaluation on hurricane damage assessment dataset.

Supports two models:
- oryx: Oryx-7B-Image model from THUdyh/Oryx-7B-Image
- videochat-flash: VideoChat-Flash-Qwen2-7B from OpenGVLab/VideoChat-Flash-Qwen2-7B_res224

Usage:
    python main.py --model oryx --test --dataset /path/to/dataset    # Test mode (2 samples)
    python main.py --model videochat-flash --dataset /path/to/images # Full run
    python main.py --model oryx --limit 50                           # Custom limit

Environment Variables:
    HURRICANE_DATASET_PATH: Path to the dataset folder containing image subfolders
    HURRICANE_LABELS_FILE: Path to the labels.json ground truth file

Dependencies:
    - See requirements.txt
    - flash-attn is NOT required (uses PyTorch's built-in SDPA attention)
"""

# =============================================================================
# Suppress warnings before importing anything else
# =============================================================================
import warnings
import logging
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress HuggingFace and transformers warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Suppress Python warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*resume_download.*')
warnings.filterwarnings('ignore', message='.*do_sample.*')
warnings.filterwarnings('ignore', message='.*weights_only.*')

# Suppress logging from various libraries
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('timm').setLevel(logging.ERROR)

import sys
import json
import argparse
import re
import torch
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Path Setup
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up to project root

# Oryx module path (in src/models/Oryx)
ORYX_DIR = os.path.join(SCRIPT_DIR, "..", "models", "Oryx")
ORYX_DIR = os.path.normpath(ORYX_DIR)
if ORYX_DIR not in sys.path:
    sys.path.insert(0, ORYX_DIR)

# Dataset paths - can be overridden via environment variables
# Default: expects a 'dataset' folder with image subfolders at runtime location
DATASET_PATH = os.environ.get("HURRICANE_DATASET_PATH", None)
if DATASET_PATH is None:
    # Default to data folder in project root (for examples) or current working directory
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset")
    if not os.path.exists(DATASET_PATH):
        DATASET_PATH = os.getcwd()

PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompt.txt")
LABELS_FILE = os.environ.get("HURRICANE_LABELS_FILE", os.path.join(PROJECT_ROOT, "data", "labels.json"))


# =============================================================================
# Helper Functions (adapted from evaluate_vlms.py)
# =============================================================================

def load_prompt_template():
    """Load the prompt template from file."""
    with open(PROMPT_FILE, 'r') as f:
        return f.read()


def get_strict_json_prompt(location_name):
    """
    Generate a stricter prompt that emphasizes JSON output.
    Used for retry mode when the model keeps returning text instead of JSON.
    """
    return f'''Analyze these two images of "{location_name}" - one showing BEFORE a hurricane, one showing AFTER.

RESPOND WITH ONLY THIS JSON FORMAT - NO OTHER TEXT:

{{
  "location_name": "{location_name}",
  "severity": "CHOOSE: Low OR Medium OR High",
  "damage_sources": ["list", "from:", "Rain", "Storm Surge", "High Wind", "Fallen Trees", "Fire", "Heavy Rain", "Lost Power"],
  "description": "2-3 sentences comparing before/after damage"
}}

Severity Guide:
- Low: Minor/cosmetic damage only
- Medium: Significant but non-structural (broken windows, fallen trees, flooding)  
- High: Major structural damage (collapsed buildings, destroyed structures)

START YOUR RESPONSE WITH {{ AND END WITH }}. NO OTHER TEXT.'''


def get_image_paths(folder_path):
    """
    Find clean and damaged image paths in a folder.
    Supports jpg, jpeg, png, bmp, webp, avif extensions.
    """
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
    """Resize image if it exceeds max dimensions while preserving aspect ratio."""
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
    """Replace the placeholder with the actual location name."""
    return template.replace("{Pass name of folder here}", location_name)


def parse_model_output(output_text, location_name):
    """
    Parse the model output to extract JSON fields.
    Returns a dictionary with location_name, severity, damage_sources, description.
    """
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
        json_str = None
        
        # Method 1: Look for ```json ... ``` blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', output_text)
        if json_match:
            json_str = json_match.group(1).strip()
        
        # Method 2: Try to find JSON object directly (starts with { and ends with })
        if not json_str:
            json_match = re.search(r'(\{[\s\S]*\})', output_text)
            if json_match:
                json_str = json_match.group(1).strip()
        
        # Method 3: If still nothing, maybe the whole output is JSON
        if not json_str and output_text.strip().startswith('{'):
            json_str = output_text.strip()
        
        if not json_str:
            return default_result
        
        # Clean up common JSON issues
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unquoted keys (simple cases)
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
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
        # Fallback: try to extract fields using regex
        result = extract_fields_with_regex(output_text, location_name)
        if not result.get("parse_error"):
            return result
        default_result["json_error"] = str(e)
        return default_result
    except Exception as e:
        default_result["error"] = str(e)
        return default_result


def extract_fields_with_regex(output_text, location_name):
    """
    Fallback extraction using regex patterns when JSON parsing fails.
    Tries to extract severity, damage_sources, and description from raw text.
    """
    result = {
        "location_name": location_name,
        "severity": "Unknown",
        "damage_sources": [],
        "description": "",
        "parse_error": True,
        "raw_output": output_text
    }
    
    # Try to extract severity from JSON-style format
    severity_match = re.search(r'"?severity"?\s*:\s*"?(Low|Medium|High)"?', output_text, re.IGNORECASE)
    if severity_match:
        result["severity"] = severity_match.group(1).title()
        result["parse_error"] = False
    
    # Try to extract damage_sources array
    sources_match = re.search(r'"?damage_sources"?\s*:\s*\[(.*?)\]', output_text, re.DOTALL)
    if sources_match:
        sources_str = sources_match.group(1)
        # Extract quoted strings from the array
        sources = re.findall(r'"([^"]+)"', sources_str)
        if sources:
            result["damage_sources"] = sources
            result["parse_error"] = False
    
    # Try to extract description
    desc_match = re.search(r'"?description"?\s*:\s*"([^"]*(?:\\.[^"]*)*)"', output_text)
    if desc_match:
        result["description"] = desc_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        result["parse_error"] = False
    
    # If still no severity, try to infer from natural language
    if result["severity"] == "Unknown":
        result = infer_from_natural_language(output_text, result)
    
    return result


def infer_from_natural_language(output_text, result):
    """
    Try to infer severity and damage sources from natural language text
    when the model doesn't return JSON.
    """
    text_lower = output_text.lower()
    
    # Severity inference based on keywords
    high_indicators = [
        'severe damage', 'major damage', 'catastrophic', 'destroyed', 'collapsed',
        'completely damaged', 'total destruction', 'heavily damaged', 'extensive damage',
        'significant structural', 'major structural', 'devastation', 'demolished',
        'beyond repair', 'uninhabitable', 'total loss'
    ]
    medium_indicators = [
        'moderate damage', 'partial damage', 'some damage', 'noticeable damage',
        'roof damage', 'broken windows', 'fallen trees', 'flooding', 'debris',
        'damaged vehicles', 'partial destruction', 'impacted', 'affected'
    ]
    low_indicators = [
        'minor damage', 'minimal damage', 'slight damage', 'cosmetic damage',
        'small branches', 'little damage', 'mostly intact', 'minor flooding',
        'superficial'
    ]
    
    # Check for severity indicators (prioritize high -> medium -> low)
    for indicator in high_indicators:
        if indicator in text_lower:
            result["severity"] = "High"
            result["parse_error"] = False
            break
    
    if result["severity"] == "Unknown":
        for indicator in medium_indicators:
            if indicator in text_lower:
                result["severity"] = "Medium"
                result["parse_error"] = False
                break
    
    if result["severity"] == "Unknown":
        for indicator in low_indicators:
            if indicator in text_lower:
                result["severity"] = "Low"
                result["parse_error"] = False
                break
    
    # Damage source inference
    damage_source_keywords = {
        "Storm Surge": ['storm surge', 'flooding', 'flooded', 'submerged', 'water damage', 'standing water'],
        "High Wind": ['wind damage', 'high wind', 'blown', 'wind-driven', 'wind torn'],
        "Fallen Trees": ['fallen tree', 'downed tree', 'toppled tree', 'uprooted', 'tree on'],
        "Rain": ['rain damage', 'water intrusion'],
        "Heavy Rain": ['heavy rain', 'rainfall', 'rain water'],
        "Fire": ['fire damage', 'burned', 'charred', 'fire'],
        "Lost Power": ['power outage', 'no power', 'downed power lines', 'electrical']
    }
    
    detected_sources = []
    for source, keywords in damage_source_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                if source not in detected_sources:
                    detected_sources.append(source)
                break
    
    if detected_sources:
        result["damage_sources"] = detected_sources
        result["parse_error"] = False
    
    # Use the raw text as description if we extracted something useful
    if result["severity"] != "Unknown" and not result["description"]:
        # Clean up the text for use as description
        # Take first 500 chars, clean it up
        desc = output_text.strip()[:500]
        # Remove common preambles
        for preamble in ["The image provided", "Based on the", "Looking at"]:
            if desc.startswith(preamble):
                desc = desc[len(preamble):].strip()
                if desc.startswith(','):
                    desc = desc[1:].strip()
        result["description"] = desc if len(desc) > 20 else output_text[:300]
    
    return result


# =============================================================================
# Oryx Model Class
# =============================================================================

class OryxModel:
    """
    Oryx-7B-Image model wrapper for hurricane damage assessment.
    Uses the local oryx module and downloads weights from THUdyh/Oryx-7B-Image.
    """
    
    def __init__(self, model_id="THUdyh/Oryx-7B-Image", device="cuda"):
        print(f"Loading Oryx model: {model_id} on {device}...")
        self.device = device
        self.model_path = model_id
        
        # Set environment variables for Oryx image processing
        os.environ['LOWRES_RESIZE'] = '384x32'
        os.environ['HIGHRES_BASE'] = '0x32'
        os.environ['MAXRES'] = '1536'
        os.environ['MINRES'] = '0'
        
        # Import Oryx components
        from transformers import AutoConfig
        from oryx.model.builder import load_pretrained_model
        from oryx.utils import disable_torch_init
        from oryx.mm_utils import get_model_name_from_path
        
        disable_torch_init()
        model_name = get_model_name_from_path(model_id)
        
        # Configure model overwrite settings
        # Use SDPA (Scaled Dot Product Attention) which is built into PyTorch 2.0+
        # This does NOT require flash-attn to be installed
        overwrite_config = {
            "mm_resampler_type": "dynamic_compressor",
            "patchify_video_feature": False,
            "attn_implementation": "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        }
        
        # Load model, tokenizer, and image processor
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_id, 
            None,  # model_base
            model_name, 
            device_map="cuda:0", 
            overwrite_config=overwrite_config
        )
        self.model.to(device).eval()
        print("Oryx model loaded successfully.")
    
    def generate(self, clean_img_path, damaged_img_path, prompt_text):
        """
        Generate damage assessment for a pair of images.
        
        Args:
            clean_img_path: Path to the clean/before image
            damaged_img_path: Path to the damaged/after image
            prompt_text: The formatted prompt text
            
        Returns:
            str: Model output text
        """
        import re as re_module
        import transformers
        from oryx.conversation import conv_templates, SeparatorStyle
        from oryx.mm_utils import process_anyres_highres_image_genli, KeywordsStoppingCriteria
        from oryx.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
        
        # Load and process images
        clean_img = Image.open(clean_img_path).convert('RGB')
        damaged_img = Image.open(damaged_img_path).convert('RGB')
        
        # Resize if needed to prevent OOM
        clean_img = resize_image_if_needed(clean_img)
        damaged_img = resize_image_if_needed(damaged_img)
        
        image_sizes = [clean_img.size, damaged_img.size]
        
        # Format question with image tokens (one per image)
        question = "<image>\n<image>\n" + prompt_text
        
        # Setup conversation
        conv_mode = "qwen_1_5"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        
        # Preprocess input using Qwen format
        input_ids = self._preprocess_qwen(
            [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': None}],
            self.tokenizer,
            has_image=True
        ).cuda()
        
        # Process both images
        self.image_processor.do_resize = False
        self.image_processor.do_center_crop = False
        
        image_tensor_list = []
        image_highres_tensor_list = []
        
        for img in [clean_img, damaged_img]:
            img_tensor, img_highres_tensor = process_anyres_highres_image_genli(img, self.image_processor)
            image_tensor_list.append(img_tensor.to(dtype=torch.bfloat16, device=self.device))
            image_highres_tensor_list.append(img_highres_tensor.to(dtype=torch.bfloat16, device=self.device))
        
        # Setup stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # Generate output
        # Note: images and images_highres must be lists of tensors, one per image
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                modalities=['image', 'image'],  # One entry per image
                images=image_tensor_list,
                images_highres=image_highres_tensor_list,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
        
        # Decode output
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs
    
    def _preprocess_qwen(self, sources, tokenizer, has_image=False, max_len=2048, 
                         system_message="You are a helpful assistant."):
        """Preprocess inputs for Qwen-style tokenization."""
        from oryx.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        
        roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        
        im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
        nl_tokens = tokenizer("\n").input_ids
        _system = tokenizer("system").input_ids + nl_tokens
        
        input_ids = []
        source = sources
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]
        
        input_id = []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
                num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
                texts = sentence["value"].split('<image>')
                _input_id = tokenizer(role).input_ids + nl_tokens 
                for i, text in enumerate(texts):
                    _input_id += tokenizer(text).input_ids 
                    if i < len(texts) - 1:
                        _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
                _input_id += [im_end] + nl_tokens
            else:
                if sentence["value"] is None:
                    _input_id = tokenizer(role).input_ids + nl_tokens
                else:
                    _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
        
        input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids


# =============================================================================
# VideoChat-Flash Model Class
# =============================================================================

class VideoChatFlashModel:
    """
    VideoChat-Flash-Qwen2-7B model wrapper for hurricane damage assessment.
    Uses the model from OpenGVLab/VideoChat-Flash-Qwen2-7B_res224.
    
    NOTE: The HuggingFace cached model files have been patched to make flash_attn
    optional. The model uses PyTorch SDPA as a fallback when flash_attn is not available.
    """
    
    def __init__(self, model_id="OpenGVLab/VideoChat-Flash-Qwen2-7B_res224", device="cuda"):
        print(f"Loading VideoChat-Flash model: {model_id}...")
        self.device = device
        self.model_path = model_id
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()
        
        self.image_processor = self.model.get_vision_tower().image_processor
        
        # Disable mm_llm_compress for simpler processing
        self.model.config.mm_llm_compress = False
        
        print("VideoChat-Flash model loaded successfully.")
    
    def generate(self, clean_img_path, damaged_img_path, prompt_text):
        """
        Generate damage assessment for a pair of images.
        Manually prepares images as video frames since the model's chat() only accepts one input.
        
        The model expects videos with at least 8 frames (mm_local_num_frames=8).
        We pad our 2 images to 8 frames by repeating them.
        
        Args:
            clean_img_path: Path to the clean/before image
            damaged_img_path: Path to the damaged/after image
            prompt_text: The formatted prompt text
            
        Returns:
            str: Model output text
        """
        from torchvision import transforms
        
        # Load images as PIL Images
        clean_img = Image.open(clean_img_path).convert('RGB')
        damaged_img = Image.open(damaged_img_path).convert('RGB')
        
        # Resize if needed to prevent OOM
        clean_img = resize_image_if_needed(clean_img)
        damaged_img = resize_image_if_needed(damaged_img)
        
        # Get image sizes (H, W format)
        image_sizes = [(224, 224)]
        
        # Manual preprocessing to match the model's expected format
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process each image
        clean_tensor = transform(clean_img)  # [C, H, W]
        damaged_tensor = transform(damaged_img)  # [C, H, W]
        
        # Model requires at least 8 frames (mm_local_num_frames=8)
        # Pad by repeating: clean x4, then damaged x4 = 8 frames
        frames_list = [clean_tensor] * 4 + [damaged_tensor] * 4
        
        # Stack into [T, C, H, W] - NO batch dimension
        # Model expects images as list of [T, C, H, W] tensors
        frames_tensor = torch.stack(frames_list, dim=0)  # [8, 3, 224, 224]
        frames_tensor = frames_tensor.to(dtype=self.model.dtype, device='cuda')
        
        # Wrap in list - each element is one "video" with shape [T, C, H, W]
        processed_frames = [frames_tensor]
        
        # Import conversation utilities from the model's module
        import importlib
        model_module = importlib.import_module(
            "transformers_modules.OpenGVLab.VideoChat-Flash-Qwen2-7B_res224.293b80d125b625c80d792fc574d67c1551a4766b.conversation"
        )
        conv_templates = model_module.conv_templates
        SeparatorStyle = model_module.SeparatorStyle
        
        mm_utils_module = importlib.import_module(
            "transformers_modules.OpenGVLab.VideoChat-Flash-Qwen2-7B_res224.293b80d125b625c80d792fc574d67c1551a4766b.mm_utils"
        )
        tokenizer_image_token = mm_utils_module.tokenizer_image_token
        KeywordsStoppingCriteria = mm_utils_module.KeywordsStoppingCriteria
        
        constants_module = importlib.import_module(
            "transformers_modules.OpenGVLab.VideoChat-Flash-Qwen2-7B_res224.293b80d125b625c80d792fc574d67c1551a4766b.constants"
        )
        IMAGE_TOKEN_INDEX = constants_module.IMAGE_TOKEN_INDEX
        DEFAULT_IMAGE_TOKEN = constants_module.DEFAULT_IMAGE_TOKEN
        
        # Build conversation
        conv = conv_templates["qwen_2"].copy()
        
        # Format prompt with image token - mention that frames 1-4 are clean, 5-8 are damaged
        user_prompt = f"{DEFAULT_IMAGE_TOKEN}\nThis video shows 8 frames: frames 1-4 show the location BEFORE the hurricane (clean/undamaged), frames 5-8 show the SAME location AFTER the hurricane (damaged). Compare the before and after states.\n\n{prompt_text}"
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        
        # Set pad token if needed
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 151643  # Qwen pad token
        
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
        
        # Setup stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=processed_frames,
                attention_mask=attention_masks,
                modalities=["video"],
                image_sizes=image_sizes,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=False,
                max_new_tokens=512,
                num_beams=1
            )
        
        # Decode output
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        
        return outputs.strip()


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def load_existing_results(output_path):
    """Load existing results file if it exists, for resuming."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_results(output_path, results):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def process_single_sample(model_engine, location_name, prompt_or_template):
    """
    Process a single sample and return the result.
    Returns (result_dict, had_parse_error, raw_output_if_error)
    
    prompt_or_template: Either a template with {Pass name of folder here} placeholder,
                        or a ready-to-use prompt string.
    """
    folder_path = os.path.join(DATASET_PATH, location_name)
    
    if not os.path.exists(folder_path):
        return None, False, f"Folder not found: {folder_path}"
    
    clean_path, damaged_path = get_image_paths(folder_path)
    
    if clean_path is None or damaged_path is None:
        return None, False, f"Images not found in {folder_path}"
    
    if not os.path.exists(clean_path) or not os.path.exists(damaged_path):
        return None, False, f"Image files missing in {folder_path}"
    
    # Check if this is a template (has placeholder) or a direct prompt
    if "{Pass name of folder here}" in prompt_or_template:
        prompt_text = format_prompt(prompt_or_template, location_name)
    else:
        prompt_text = prompt_or_template
    
    try:
        output = model_engine.generate(clean_path, damaged_path, prompt_text)
        prediction = parse_model_output(output, location_name)
        
        result_item = {
            "location_name": location_name,
            "severity": prediction.get("severity", "Unknown"),
            "damage_sources": prediction.get("damage_sources", []),
            "description": prediction.get("description", "")
        }
        
        had_parse_error = prediction.get("parse_error", False)
        raw_output = prediction.get("raw_output", "") if had_parse_error else ""
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return result_item, had_parse_error, raw_output
        
    except Exception as e:
        return {
            "location_name": location_name,
            "severity": "Unknown",
            "damage_sources": [],
            "description": ""
        }, True, str(e)


def run_single_model(model_name, device, prompt_template, labels_data, limit, no_retry):
    """
    Run evaluation for a single model.
    Returns the number of samples processed.
    """
    # Determine output filename
    if model_name == "oryx":
        output_filename = "labels_oryx.json"
    elif model_name == "videochat-flash":
        output_filename = "labels_videochat-flash-qwen2.json"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    output_path = os.path.join(DATASET_PATH, output_filename)
    
    # Load existing results for resume support
    existing_results = load_existing_results(output_path)
    processed_locations = {r["location_name"] for r in existing_results}
    
    # Build results dict for easy updating
    results_dict = {r["location_name"]: r for r in existing_results}
    
    # Count how many we need to process
    to_process = []
    for item in labels_data:
        if len(to_process) >= limit:
            break
        location_name = item['location_name']
        if location_name not in processed_locations:
            to_process.append(location_name)
    
    if not to_process:
        print(f"All samples already processed for {model_name}. Found {len(existing_results)} existing results.")
        if no_retry:
            return len(existing_results)
    else:
        print(f"Found {len(existing_results)} existing results. Processing {len(to_process)} remaining samples...")
    
    # Initialize model
    model_engine = None
    
    if model_name == "oryx":
        model_engine = OryxModel(device=device)
    elif model_name == "videochat-flash":
        model_engine = VideoChatFlashModel(device=device)
    
    if model_engine is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Track failed samples for retry
    failed_locations = []
    
    # Process new samples
    if to_process:
        for location_name in tqdm(to_process, desc=f"Evaluating {model_name}"):
            result, had_error, error_info = process_single_sample(
                model_engine, location_name, prompt_template
            )
            
            if result is None:
                print(f"\nSkipping {location_name}: {error_info}")
                continue
            
            results_dict[location_name] = result
            
            if had_error:
                failed_locations.append(location_name)
                snippet = error_info[:150] if error_info else "Unknown error"
                print(f"\nWarning: Failed to parse JSON for {location_name}")
                print(f"  Snippet: {snippet}...")
            
            # Save after each sample (incremental save)
            results_list = list(results_dict.values())
            save_results(output_path, results_list)
    
    # Retry failed samples
    if not no_retry and failed_locations:
        print(f"\n{'='*60}")
        print(f"Retrying {len(failed_locations)} failed samples...")
        print(f"{'='*60}")
        
        retry_success = 0
        for location_name in tqdm(failed_locations, desc="Retrying failed"):
            result, had_error, error_info = process_single_sample(
                model_engine, location_name, prompt_template
            )
            
            if result is None:
                continue
            
            if not had_error:
                # Success on retry!
                results_dict[location_name] = result
                retry_success += 1
                print(f"\nRetry SUCCESS: {location_name}")
            else:
                # Still failed, keep the result anyway
                results_dict[location_name] = result
            
            # Save after each retry
            results_list = list(results_dict.values())
            save_results(output_path, results_list)
        
        print(f"\nRetry complete: {retry_success}/{len(failed_locations)} samples recovered")
    
    # Final save
    results_list = list(results_dict.values())
    save_results(output_path, results_list)
    
    # Summary
    total_unknown = sum(1 for r in results_list if r.get("severity") == "Unknown")
    print(f"\n{'='*60}")
    print(f"Processing complete for {model_name}. Results saved to {output_path}")
    print(f"Total samples in output: {len(results_list)}")
    print(f"Samples with Unknown severity: {total_unknown}")
    print(f"{'='*60}")
    
    # Clean up model to free GPU memory before next model
    del model_engine
    torch.cuda.empty_cache()
    
    return len(results_list)


def is_failed_entry(entry):
    """
    Check if an entry should be considered 'failed' and needs re-processing.
    Returns True if severity is Unknown OR description is empty/missing.
    """
    severity = entry.get("severity", "Unknown")
    description = entry.get("description", "")
    damage_sources = entry.get("damage_sources", [])
    
    # Consider failed if:
    # - Severity is Unknown
    # - Description is empty or very short (less than 20 chars)
    # - No damage sources identified
    if severity == "Unknown":
        return True
    if not description or len(description.strip()) < 20:
        return True
    if not damage_sources:
        return True
    
    return False


def retry_failed_entries(model_name, device, prompt_template):
    """
    Load existing results and retry only the failed entries.
    """
    # Determine output filename
    if model_name == "oryx":
        output_filename = "labels_oryx.json"
    elif model_name == "videochat-flash":
        output_filename = "labels_videochat-flash-qwen2.json"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    output_path = os.path.join(DATASET_PATH, output_filename)
    
    # Load existing results
    if not os.path.exists(output_path):
        print(f"No existing results file found: {output_path}")
        print("Run normal evaluation first.")
        return 0
    
    existing_results = load_existing_results(output_path)
    if not existing_results:
        print("No existing results to retry.")
        return 0
    
    # Find failed entries
    failed_entries = []
    for entry in existing_results:
        if is_failed_entry(entry):
            failed_entries.append(entry["location_name"])
    
    if not failed_entries:
        print(f"No failed entries found in {output_filename}!")
        print(f"All {len(existing_results)} entries have valid severity, description, and damage sources.")
        return 0
    
    print(f"Found {len(failed_entries)} failed entries out of {len(existing_results)} total.")
    print(f"Failed locations: {failed_entries[:10]}{'...' if len(failed_entries) > 10 else ''}")
    print()
    
    # Build results dict for easy updating
    results_dict = {r["location_name"]: r for r in existing_results}
    
    # Initialize model
    print(f"Loading {model_name} model for retry...")
    if model_name == "oryx":
        model_engine = OryxModel(device=device)
    elif model_name == "videochat-flash":
        model_engine = VideoChatFlashModel(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Process failed entries with strict JSON prompt
    success_count = 0
    still_failed = []
    
    for location_name in tqdm(failed_entries, desc=f"Retrying failed ({model_name})"):
        # Use strict JSON prompt for retry
        strict_prompt = get_strict_json_prompt(location_name)
        result, had_error, error_info = process_single_sample(
            model_engine, location_name, strict_prompt
        )
        
        if result is None:
            print(f"\nSkipping {location_name}: {error_info}")
            still_failed.append(location_name)
            continue
        
        # Check if the retry was successful
        if not is_failed_entry(result):
            results_dict[location_name] = result
            success_count += 1
            print(f"\nFixed: {location_name} -> severity={result['severity']}")
        else:
            # Still failed, but update with new attempt anyway
            results_dict[location_name] = result
            still_failed.append(location_name)
            if had_error:
                snippet = error_info[:100] if error_info else "Unknown"
                print(f"\nStill failed: {location_name} ({snippet}...)")
        
        # Save after each retry
        results_list = list(results_dict.values())
        save_results(output_path, results_list)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"RETRY COMPLETE for {model_name}")
    print(f"{'='*60}")
    print(f"Successfully fixed: {success_count}/{len(failed_entries)}")
    print(f"Still failing: {len(still_failed)}")
    
    if still_failed:
        print(f"\nStill failing locations:")
        for loc in still_failed[:20]:
            entry = results_dict.get(loc, {})
            print(f"  - {loc}: severity={entry.get('severity', 'N/A')}, desc_len={len(entry.get('description', ''))}")
        if len(still_failed) > 20:
            print(f"  ... and {len(still_failed) - 20} more")
    
    # Clean up
    del model_engine
    torch.cuda.empty_cache()
    
    return success_count


def main():
    global DATASET_PATH, LABELS_FILE
    
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Hurricane Damage Assessment Dataset")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["oryx", "videochat-flash", "both-zachary"],
                        help="Model to use for evaluation (both-zachary runs oryx then videochat-flash)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset folder containing image subfolders (overrides HURRICANE_DATASET_PATH)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels.json ground truth file (overrides HURRICANE_LABELS_FILE)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: only process 2 samples")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--no-retry", action="store_true",
                        help="Skip retrying failed samples at the end")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only retry entries that previously failed (Unknown severity, empty description, etc.)")
    args = parser.parse_args()
    
    # Override paths if provided via command line
    if args.dataset:
        DATASET_PATH = args.dataset
    if args.labels:
        LABELS_FILE = args.labels
    
    # Check for GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. No GPU detected.")
    device = "cuda"
    
    # Load prompt template
    prompt_template = load_prompt_template()
    
    # Handle --retry-failed mode
    if args.retry_failed:
        print("\n" + "="*60)
        print("RETRY-FAILED MODE")
        print("Re-processing entries with Unknown severity or empty fields")
        print("="*60 + "\n")
        
        if args.model == "both-zachary":
            models_to_run = ["oryx", "videochat-flash"]
        else:
            models_to_run = [args.model]
        
        total_fixed = 0
        for model_name in models_to_run:
            print(f"\n{'#'*60}")
            print(f"# RETRYING: {model_name.upper()}")
            print(f"{'#'*60}\n")
            fixed = retry_failed_entries(model_name, device, prompt_template)
            total_fixed += fixed
        
        print(f"\n{'='*60}")
        print(f"RETRY-FAILED COMPLETE. Total fixed: {total_fixed}")
        print("="*60)
        return
    
    # Normal processing mode
    # Load ground truth labels to get list of locations
    with open(LABELS_FILE, 'r') as f:
        labels_data = json.load(f)
    
    # Determine sample limit
    if args.test:
        limit = 2
        print("Test mode: Processing 2 samples only")
    elif args.limit:
        limit = args.limit
        print(f"Processing up to {limit} samples")
    else:
        limit = len(labels_data)
        print(f"Processing all {len(labels_data)} samples")
    
    # Determine which models to run
    if args.model == "both-zachary":
        models_to_run = ["oryx", "videochat-flash"]
        print("\n" + "="*60)
        print("RUNNING BOTH MODELS (Zachary mode)")
        print("Order: Oryx -> VideoChat-Flash")
        print("="*60 + "\n")
    else:
        models_to_run = [args.model]
    
    # Run each model
    for i, model_name in enumerate(models_to_run):
        if len(models_to_run) > 1:
            print(f"\n{'#'*60}")
            print(f"# MODEL {i+1}/{len(models_to_run)}: {model_name.upper()}")
            print(f"{'#'*60}\n")
        
        run_single_model(
            model_name=model_name,
            device=device,
            prompt_template=prompt_template,
            labels_data=labels_data,
            limit=limit,
            no_retry=args.no_retry
        )
    
    if len(models_to_run) > 1:
        print("\n" + "="*60)
        print("ALL MODELS COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    main()

