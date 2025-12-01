import json
import sys
import os
from typing import List, Dict, Any
import subprocess

# Check and install required packages
def install_packages():
    packages = ['nltk', 'rouge-score']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Package '{package}' not found")
            sys.exit(1)
install_packages()

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# =============================================================================
# Severity Score
# =============================================================================

SEVERITY_LEVELS = ["Low", "Medium", "High"]
SEVERITY_MAP = {"Low": 0, "Medium": 1, "High": 2}

def compute_severity_score(ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
    exact_matches = 0
    weighted_score = 0.0
    total_valid = 0
    
    gt_map = {item["location_name"]: item for item in ground_truth}
    
    for pred_item in predictions:
        loc = pred_item.get("location_name", "")
        gt_item = gt_map.get(loc)
        
        if not gt_item:
            continue
        
        gt = gt_item.get("severity")
        pred = pred_item.get("severity")

        if gt not in SEVERITY_LEVELS:
            continue
            
        total_valid += 1
        
        if gt == pred:
            exact_matches += 1
            weighted_score += 1.0
        elif pred in SEVERITY_LEVELS:
            gt_idx = SEVERITY_MAP[gt]
            pred_idx = SEVERITY_MAP[pred]
            distance = abs(gt_idx - pred_idx)
            weighted_score += 1.0 - (distance * 0.5)
    
    if total_valid == 0:
        return {"error": "No valid severity labels found"}
    
    return {
        "exact_accuracy": exact_matches / total_valid,
        "weighted_accuracy": weighted_score / total_valid,
        "total_samples": total_valid
    }

# =============================================================================
# Damage Source Score
# =============================================================================

ALL_DAMAGE_SOURCES = [
    "Rain",
    "Storm Surge",
    "High Wind",
    "Fallen Trees",
    "Fire",
    "Heavy Rain",
    "Lost Power"
]

def compute_source_score(ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_source_tp = {s: 0 for s in ALL_DAMAGE_SOURCES}
    per_source_fp = {s: 0 for s in ALL_DAMAGE_SOURCES}
    per_source_fn = {s: 0 for s in ALL_DAMAGE_SOURCES}
    
    total_valid = 0
    gt_map = {item["location_name"]: item for item in ground_truth}
    
    for pred_item in predictions:
        loc = pred_item.get("location_name", "")
        gt_item = gt_map.get(loc)
        
        if not gt_item:
            continue
        
        gt_sources = set(gt_item.get("damage_sources", []))
        pred_sources = set(pred_item.get("damage_sources", []))
        
        gt_sources_normalized = {s.strip().title() for s in gt_sources if s}
        pred_sources_normalized = {s.strip().title() for s in pred_sources if s}
        
        if not gt_sources_normalized:
            continue
            
        total_valid += 1
        
        tp = len(gt_sources_normalized & pred_sources_normalized)
        fp = len(pred_sources_normalized - gt_sources_normalized)
        fn = len(gt_sources_normalized - pred_sources_normalized)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        for source in ALL_DAMAGE_SOURCES:
            source_normalized = source.strip().title()
            in_gt = source_normalized in gt_sources_normalized
            in_pred = source_normalized in pred_sources_normalized
            
            if in_gt and in_pred:
                per_source_tp[source] += 1
            elif in_pred and not in_gt:
                per_source_fp[source] += 1
            elif in_gt and not in_pred:
                per_source_fn[source] += 1
    
    if total_valid == 0:
        return {"error": "No valid damage source labels found"}
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    f1_scores = []
    for source in ALL_DAMAGE_SOURCES:
        tp = per_source_tp[source]
        fp = per_source_fp[source]
        fn = per_source_fn[source]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if tp + fn > 0:
            f1_scores.append(f1)
    
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "total_samples": total_valid
    }


# =============================================================================
# BLEU and ROUGE Score
# =============================================================================

def compute_bleu_score(reference: str, hypothesis: str) -> Dict[str, float]:
    if not reference or not hypothesis:
        return {"bleu_1": 0, "bleu_2": 0, "bleu_3": 0, "bleu_4": 0}
    
    try:
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    except:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
    
    if not ref_tokens or not hyp_tokens:
        return {"bleu_1": 0, "bleu_2": 0, "bleu_3": 0, "bleu_4": 0}
    
    smoothie = SmoothingFunction().method1
    
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1/n] * n + [0] * (4-n))
        try:
            score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothie)
        except:
            score = 0
        bleu_scores[f"bleu_{n}"] = score
    
    return bleu_scores


def compute_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    if not reference or not hypothesis:
        return {"rouge_1": 0, "rouge_2": 0, "rouge_l": 0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    
    return {
        "rouge_1": scores['rouge1'].fmeasure,
        "rouge_2": scores['rouge2'].fmeasure,
        "rouge_l": scores['rougeL'].fmeasure
    }


def compute_description_metrics(ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
    bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    rouge_scores = {"rouge_1": [], "rouge_2": [], "rouge_l": []}
    
    total_valid = 0
    gt_map = {item["location_name"]: item for item in ground_truth}
    
    for pred_item in predictions:
        loc = pred_item.get("location_name", "")
        gt_item = gt_map.get(loc)
        
        if not gt_item:
            continue
        
        gt_desc = gt_item.get("description", "")
        pred_desc = pred_item.get("description", "")
        
        if not gt_desc or "MISSING" in gt_desc or not pred_desc:
            continue
            
        total_valid += 1
        
        bleu = compute_bleu_score(gt_desc, pred_desc)
        for key in bleu_scores:
            bleu_scores[key].append(bleu[key])
        
        rouge = compute_rouge_scores(gt_desc, pred_desc)
        for key in rouge_scores:
            rouge_scores[key].append(rouge[key])
    
    if total_valid == 0:
        return {"error": "No valid descriptions found"}
    
    result = {"total_samples": total_valid}
    
    for key, scores in bleu_scores.items():
        result[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0
    
    for key, scores in rouge_scores.items():
        result[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0
    
    return result


# =============================================================================
# SPICE-LIKE SEMANTIC METRICS
# =============================================================================

DAMAGE_OBJECTS = [
    "roof", "roofs", "roofing", "shingles",
    "window", "windows", "glass",
    "wall", "walls", "facade", "structure", "building", "buildings", "house", "houses", "home", "homes",
    "tree", "trees", "vegetation", "palm", "palms", "foliage",
    "road", "roads", "street", "streets", "pavement", "highway", "bridge",
    "car", "cars", "vehicle", "vehicles", "boat", "boats",
    "power", "powerlines", "utility", "poles", "electrical",
    "debris", "rubble", "wreckage",
    "water", "flood", "flooding", "floodwater", "surge",
    "sand", "mud", "silt", "sediment"
]

DAMAGE_ATTRIBUTES = [
    "damaged", "destroyed", "collapsed", "broken", "shattered", "missing",
    "flooded", "submerged", "inundated", "underwater",
    "fallen", "toppled", "uprooted", "snapped", "bent",
    "stripped", "torn", "ripped", "peeled",
    "scattered", "displaced", "overturned",
    "intact", "undamaged", "standing",
    "severe", "major", "minor", "catastrophic", "devastating", "extensive"
]


def extract_semantic_propositions(text: str) -> Dict[str, set]:
    if not text:
        return {"objects": set(), "attributes": set(), "pairs": set()}
    
    text_lower = text.lower()
    
    try:
        tokens = nltk.word_tokenize(text_lower)
    except:
        tokens = text_lower.split()
    
    found_objects = set()
    found_attributes = set()
    found_pairs = set()
    
    for obj in DAMAGE_OBJECTS:
        if obj in text_lower:
            found_objects.add(obj)
    
    for attr in DAMAGE_ATTRIBUTES:
        if attr in text_lower:
            found_attributes.add(attr)
    
    sentences = text_lower.replace('!', '.').replace('?', '.').split('.')
    for sentence in sentences:
        sentence_objects = [obj for obj in DAMAGE_OBJECTS if obj in sentence]
        sentence_attrs = [attr for attr in DAMAGE_ATTRIBUTES if attr in sentence]
        
        for obj in sentence_objects:
            for attr in sentence_attrs:
                found_pairs.add((attr, obj))
    
    return {
        "objects": found_objects,
        "attributes": found_attributes,
        "pairs": found_pairs
    }


def compute_spice_score(reference: str, hypothesis: str) -> Dict[str, float]:
    ref_props = extract_semantic_propositions(reference)
    hyp_props = extract_semantic_propositions(hypothesis)
    
    metrics = {}
    
    for prop_type in ["objects", "attributes", "pairs"]:
        ref_set = ref_props[prop_type]
        hyp_set = hyp_props[prop_type]
        
        if not ref_set:
            metrics[f"{prop_type}_f1"] = 1.0 if not hyp_set else 0.0
            continue
        
        tp = len(ref_set & hyp_set)
        precision = tp / len(hyp_set) if hyp_set else 0
        recall = tp / len(ref_set) if ref_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f"{prop_type}_f1"] = f1
    
    combined_f1 = (
        0.3 * metrics["objects_f1"] + 
        0.3 * metrics["attributes_f1"] + 
        0.4 * metrics["pairs_f1"]
    )
    metrics["spice_score"] = combined_f1
    
    return metrics


def compute_semantic_metrics(ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
    all_metrics = {
        "objects_f1": [], "attributes_f1": [], "pairs_f1": [], "spice_score": []
    }
    
    total_valid = 0
    gt_map = {item["location_name"]: item for item in ground_truth}
    
    for pred_item in predictions:
        loc = pred_item.get("location_name", "")
        gt_item = gt_map.get(loc)
        
        if not gt_item:
            continue
        
        gt_desc = gt_item.get("description", "")
        pred_desc = pred_item.get("description", "")
        
        if not gt_desc or "MISSING" in gt_desc or not pred_desc:
            continue
            
        total_valid += 1
        
        metrics = compute_spice_score(gt_desc, pred_desc)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    if total_valid == 0:
        return {"error": "No valid descriptions for semantic analysis"}
    
    result = {"total_samples": total_valid}
    for key, scores in all_metrics.items():
        result[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0
    
    return result

def evaluate_results(model_path: str, labels_path: str = "labels.json") -> Dict[str, Any]:    
    # Load ground truth labels
    with open(labels_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Load model predictions
    with open(model_path, 'r') as f:
        predictions = json.load(f)

    model_name = os.path.basename(model_path).replace("labels_", "").replace(".json", "")

    evaluation = {}
    
    # Severity score
    print("--- Severity Score ---")
    severity_metrics = compute_severity_score(ground_truth, predictions)
    evaluation["severity"] = severity_metrics
    if "error" not in severity_metrics:
        print(f"Exact Accuracy: {severity_metrics['exact_accuracy']:.3f}")
        print(f"Weighted Accuracy: {severity_metrics['weighted_accuracy']:.3f}")
        print(f"Samples: {severity_metrics['total_samples']}")
    else:
        print(f"Error: {severity_metrics['error']}")
    
    # Damage source score
    print("--- Damage Source Score ---")
    source_metrics = compute_source_score(ground_truth, predictions)
    evaluation["damage_sources"] = source_metrics
    if "error" not in source_metrics:
        print(f"Micro F1: {source_metrics['micro_f1']:.3f}")
        print(f"Micro Precision: {source_metrics['micro_precision']:.3f}")
        print(f"Micro Recall: {source_metrics['micro_recall']:.3f}")
        print(f"Macro F1: {source_metrics['macro_f1']:.3f}")
        print(f"Samples: {source_metrics['total_samples']}")
    else:
        print(f"Error: {source_metrics['error']}")
    
    # Description Score (BLEU/ROUGE)
    print("--- Description Quality (BLEU/ROUGE) ---")
    desc_metrics = compute_description_metrics(ground_truth, predictions)
    evaluation["description_bleu_rouge"] = desc_metrics
    if "error" not in desc_metrics:
        print(f"BLEU-1: {desc_metrics['avg_bleu_1']:.3f}")
        print(f"BLEU-4: {desc_metrics['avg_bleu_4']:.3f}")
        print(f"ROUGE-1: {desc_metrics['avg_rouge_1']:.3f}")
        print(f"ROUGE-L: {desc_metrics['avg_rouge_l']:.3f}")
        print(f"Samples: {desc_metrics['total_samples']}")
    else:
        print(f"Error: {desc_metrics['error']}")
    
    # Semantic metrics (SPICE-like)
    print("--- Semantic Evaluation (SPICE-like) ---")
    semantic_metrics = compute_semantic_metrics(ground_truth, predictions)
    evaluation["semantic"] = semantic_metrics
    if "error" not in semantic_metrics:
        print(f"SPICE Score: {semantic_metrics['avg_spice_score']:.3f}")
        print(f"Object F1: {semantic_metrics['avg_objects_f1']:.3f}")
        print(f"Attribute F1: {semantic_metrics['avg_attributes_f1']:.3f}")
        print(f"Pair F1: {semantic_metrics['avg_pairs_f1']:.3f}")
        print(f"Samples: {semantic_metrics['total_samples']}")
    else:
        print(f"Error: {semantic_metrics['error']}")
    
    return evaluation


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_metrics_simple.py <labels_[model].json> [labels.json]")
        sys.exit(1)
    
    results_path = sys.argv[1]
    labels_path = sys.argv[2] if len(sys.argv) > 2 else "labels.json"
    
    if not os.path.exists(labels_path):
        print(f"Error: Ground truth file '{labels_path}' not found in current directory")
        sys.exit(1)
    
    evaluate_results(results_path, labels_path)
