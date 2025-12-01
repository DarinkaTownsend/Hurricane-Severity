# 🌪️ Hurricane Damage Assessment with Vision--Language Models

Automatic comparison of before/after hurricane images using multimodal
AI

------------------------------------------------------------------------

## 📌 Overview

This project presents a large-scale evaluation of Vision--Language
Models (VLMs) for post-hurricane damage assessment using before/after
(clean/damage) image pairs. A ground truth dataset of approximately
1,000 image pairs was constructed using images collected from:

-   United States\
-   Mexico\
-   Puerto Rico\
-   Cuba

Using a unified prompting strategy, expert multimodal LLMs generate
severity labels, damage sources, and textual descriptions. Multiple VLMs
are then evaluated and compared using classification, language, and
semantic metrics.

------------------------------------------------------------------------

## 🖼️ Example of Before/After Hurricane Damage Pairs

![Sample before/after hurricane
damage](03_readme_figures/Clean.png)

------------------------------------------------------------------------

## 🧠 Evaluated Vision--Language Models

-   Qwen2.5-VL-7B-Instruct\
-   LLaVA-OneVision-Qwen2-7B\
-   Ovis2-8B\
-   InternVideo2.5_Chat-8B\
-   Oryx-7B\
-   Valley-Eagle-7B\
-   LLaVA-Video-7B-Qwen2\
-   DeepSeek-VL

Each model performs: - Damage severity classification\
- Damage source identification\
- Textual description generation

------------------------------------------------------------------------

## 🏗️ Two-Stage Methodology

1. Data Collection & Ground Truth Construction 
2. Vision--Language Model Evaluation & Comparison

------------------------------------------------------------------------

## 📊 Ground Truth Construction Pipeline

![Ground truth construction pipeline](03_readme_figures/pipeline2.png)

Key steps: 1. Damage taxonomy definition\
2. Before/After image collection\
3. Unified prompt design\
4. Automatic labeling with Gemini & GPT-5.1\
5. Label review & cleaning\
6. Final dataset construction as labels.json

------------------------------------------------------------------------

## 🤖 Model Evaluation Pipeline

![Model evaluation pipeline](03_readme_figures/pipeline3.png)

Key steps: 1. Model selection\
2. Inference over clean/damage image pairs\
3. Prediction file generation (labels\_\[model\].json)\
4. Automatic metric computation\
5. Cross-model performance comparison

------------------------------------------------------------------------

## 📐 Evaluation Metrics

Classification Metrics: - Severity Exact Accuracy\
- Severity Weighted Accuracy\
- Damage Source Micro / Macro F1-score

Language Metrics: - BLEU-1 to BLEU-4\
- ROUGE-1, ROUGE-2, ROUGE-L

Semantic Metrics (SPICE-like): - Objects F1\
- Attributes F1\
- Pairs F1\
- Global SPICE Score
