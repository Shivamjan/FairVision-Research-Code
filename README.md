# CARE-Net: Counterfactual Attention Regularized Fair and Text-Free Dermatology Diagnosis

> A fairness-aware deep learning framework for dermatological skin condition diagnosis that requires **no clinical text**, uses **counterfactual attention regularization** to reduce demographic bias, and is evaluated across diverse skin tones.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Reproducing Results](#reproducing-results)
- [Fairness Evaluation](#fairness-evaluation)
- [License](#license)

---

## Overview

Existing dermatology AI systems often rely on clinical metadata or text annotations, and frequently exhibit performance disparities across patients with different skin tones. **CARE-Net** addresses both problems simultaneously:

- It operates in a **text-free** setting — no clinical notes or metadata required, only dermoscopy images.
- It applies **counterfactual attention regularization** to encourage the model to focus on disease-relevant regions rather than skin-tone-correlated features, improving fairness across demographic groups.

CARE-Net is evaluated on **Fitzpatrick17k** and **DDI**, two benchmarks that explicitly include skin tone diversity, enabling rigorous fairness assessment.

---

## Repository Structure

```
FairVision-Research-Code/
├── pre-processings/         # Dataset loading, skin tone labeling, counterfactual augmentation
├── models/                  # CARE-Net architecture and training scripts
├── evaluations/             # Fairness metrics, result analysis, and visualization notebooks
├── LICENSE                  # MIT License
└── README.md
```

---

## Datasets

CARE-Net is trained and evaluated on two publicly available dermatology datasets:

### Fitzpatrick17k
- ~17,000 clinical skin disease images labeled with **Fitzpatrick skin type** (I–VI)
- Covers a wide range of dermatological conditions across skin tones
- Download: [link](https://github.com/mattgroh/fitzpatrick17k)

### DDI (Diverse Dermatology Images)
- A curated dataset designed to benchmark **skin tone diversity** in dermatology AI
- Includes images spanning light to dark skin tones with condition labels
- Download: [link](https://ddi-dataset.github.io/index.html#dataset)
  

### Data Directory Layout

Once downloaded, place datasets under a `data/` directory before running preprocessing:

```
data/
├── fitzpatrick17k/
│   ├── images/
│   └── fitzpatrick17k.csv
└── ddi/
    ├── images/
    └── ddi_metadata.csv
```


## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended)
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/Shivamjan/FairVision-Research-Code.git
cd FairVision-Research-Code

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Results 

### Step 1 — Preprocess the Data

This section describes the preprocessing steps required to reproduce the results of the project. The preprocessing pipeline prepares the datasets, generates textual descriptions and counterfactual descriptions, and computes text embeddings for downstream analysis.


The pipeline consists of **three main stages**:

1. Cleaning metadata for training
2. Generating text and their counterfactual 
3. Creating text embeddings

Each stage must be executed **in the specified order**:

1. `pre-processings/Cleaning Metadata for Training/clean_fitz.ipynb` — clean and standardize metadata for the Fitzpatrick17k dataset  
2. `pre-processings/Cleaning Metadata for Training/clean_ddi.ipynb` — clean and align metadata for the DDI dataset  
3. `pre-processings/Generating Texts and Their Counterfactuals/generating text.py` — generate textual descriptions for each image  
4. `pre-processings/Generating Texts and Their Counterfactuals/generating counterfactuals.py` — create counterfactual prompts by swapping skin tone attributes  
5. `pre-processings/Text Embeddings/getting embeddings.ipynb` — generate embeddings for text prompts and counterfactuals

### Step 2 — Train CARE-Net

Two variants of CARE-Net are provided:

- **PL-CQC**   
- **AC-CQC**

Navigate to the training directory:

```bash
cd models/

# Train AC-CQC
python AC-CQC/train_ac_cqc.py \
  --dataset_name fitzpatrick \
  --model_name ac_cqc \
  --metadata_csv /path/to/metadata.csv \
  --emb_orig /path/to/embeddings/original_text_emb.npy \
  --emb_cf /path/to/embeddings/counterfactual_text_emb.npy \
  --emb_names /path/to/embeddings/emb_names.npy \
  --holdout random_holdout \
  --n_epochs 20 \
  --seed 64

# Train PL-CQC
python PL-CQC/train_pl_cqc.py \
  --dataset_name fitzpatrick \
  --model_name pl_cqc \
  --metadata_csv /path/to/metadata.csv \
  --emb_orig /path/to/embeddings/original_text_emb.npy \
  --emb_cf /path/to/embeddings/counterfactual_text_emb.npy \
  --emb_names /path/to/embeddings/emb_names.npy \
  --holdout random_holdout \
  --n_epochs 20 \
  --seed 64
```

### Step 3 — Evaluate Fairness

```bash
cd evaluations/
multi_eval.py```
```
Evaluation script cover:
- Per-skin-tone accuracy breakdown
- Fairness metrics: demographic parity gap, equality of oppurtunity, accuracy disparity
  
---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions about the research, please open an [issue](https://github.com/Shivamjan/FairVision-Research-Code/issues) or reach out via GitHub: [@Shivamjan](https://github.com/Shivamjan)

---

*Paper submission in progress.*

