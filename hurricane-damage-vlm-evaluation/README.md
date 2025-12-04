# Hurricane Damage VLM Evaluation

Evaluates Vision-Language Models (VLMs) on hurricane damage assessment using before/after image pairs.

## Models Supported

- **Oryx** (`THUdyh/Oryx-7B-Image`)
- **VideoChat-Flash** (`OpenGVLab/VideoChat-Flash-Qwen2-7B_res224`)

## Requirements

- Python 3.10+
- CUDA GPU (required)
- ~16GB+ VRAM recommended

## Installation

```bash
pip install -r requirements.txt

# Install Oryx model locally
pip install -e src/models/Oryx/
```

## Dataset Structure

Your dataset should have this structure:
```
dataset_folder/
    location_name_1/
        clean.jpg      # Before hurricane
        damaged.jpg    # After hurricane
    location_name_2/
        clean.png
        damaged.png
    ...
```

## Running Evaluation

### Basic Usage

```bash
cd src/evaluation

# Run with Oryx model
python main.py --model oryx --dataset /path/to/images --labels /path/to/labels.json

# Run with VideoChat-Flash model
python main.py --model videochat-flash --dataset /path/to/images --labels /path/to/labels.json

# Run both models sequentially
python main.py --model both-zachary --dataset /path/to/images --labels /path/to/labels.json
```

### Test Mode (2 samples only)

```bash
python main.py --model oryx --dataset /path/to/images --labels /path/to/labels.json --test
```

### Limit Number of Samples

```bash
python main.py --model oryx --dataset /path/to/images --labels /path/to/labels.json --limit 50
```

## Computing Metrics

Compute metrics comparing model predictions against ground truth labels:

```bash
cd src/evaluation

# Oryx metrics
python compute_metrics.py ../../data/results/labels_oryx.json ../../data/labels.json

# VideoChat-Flash metrics
python compute_metrics.py ../../data/results/labels_videochat-flash-qwen2.json ../../data/labels.json
```

Metrics computed:
- **Severity**: Exact accuracy, weighted accuracy
- **Damage Sources**: Micro/Macro F1, precision, recall
- **Description**: BLEU-1/4, ROUGE-1/L, SPICE-like semantic scores

## Project Structure

```
hurricane-damage-vlm-evaluation/
    data/
        labels.json                          # Ground truth labels
        results/
            labels_oryx.json                 # Oryx model predictions
            labels_videochat-flash-qwen2.json # VideoChat-Flash predictions
        examples/                            # Sample images for testing
    src/
        evaluation/
            main.py                          # Main evaluation script
            compute_metrics.py               # Metrics computation
            prompt.txt                       # Evaluation prompt
        models/
            Oryx/                            # Oryx model code
```

## Example

Test with included examples:

```bash
cd src/evaluation
python main.py --model oryx --dataset ../../data/examples --labels ../../data/examples/example_labels.json --test
```
