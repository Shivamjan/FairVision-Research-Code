import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any
import random
import sys

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("Error: sentence_transformers not installed. Run: pip install sentence-transformers")
    sys.exit()


# -------------------------
# Configuration
# -------------------------
CSV_PATH = Path("path/to/your/csv")
OUT_JSON = Path("path/to/output/json")
MODEL_ID = "BioMistral/BioMistral-7B"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
INPUT_JSON = Path("path/to/previously/generated/medical/notes/json")
MAX_RETRIES = 5
TEMPERATURE = 0.5


# -------------------------
# Quantization config
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# -------------------------
# Model loading
# -------------------------
def load_biomistral_model():
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=False
        )
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Ensure you have required libraries (transformers, accelerate, bitsandbytes) installed and sufficient resources.")
        return None, None


def load_embedding_model():
    print(f"Loading Embedding Model ({EMBEDDING_MODEL_ID})...")
    # We load this to GPU ("cuda") alongside BioMistral
    # Since it is only 300M params, it won't cause OOM.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    embed_model = SentenceTransformer(EMBEDDING_MODEL_ID).to(device)
    print("Embedding model loaded.")
    return embed_model


# -------------------------
# Utilities
# -------------------------
def load_existing() -> Dict[str, Any]:
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
        except json.JSONDecodeError:
            print("Warning: existing JSON is invalid — starting fresh.")
    return {}


def get_similarity_score(text1: str, text2: str, embed_model) -> float:
    """
    Calculates cosine similarity using the Gemma-300M model.
    """
    # Encode both texts into vectors
    # convert_to_tensor=True puts them on the GPU
    emb1 = embed_model.encode(text1, convert_to_tensor=True)
    emb2 = embed_model.encode(text2, convert_to_tensor=True)

    # Calculate cosine similarity
    score = util.cos_sim(emb1, emb2)

    # .item() extracts the float value from the tensor
    return score.item()


def save_results(data: Dict[str, Any]):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def clean_text(t: str) -> str:
    return t.replace("</s>", "").strip()

def embed(text: str) -> np.ndarray:
    raise NotImplementedError("Plug in your sentence embedding model here")

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_assistant_text(generated_text: str) -> str:
    if "[/INST]" in generated_text:
        return generated_text.split("[/INST]", 1)[-1].strip()
    return generated_text.strip()


#--------------------------------------
# GETTING FITZPATRICK COUNTERFACTUAL
#--------------------------------------
FITZPATRICK_PROFILES = {
    1: {
        "id": "I",
        "adjectives": "pale white, fair, ivory",
        "reaction": "always burns easily, never tans, highly sun-sensitive"
    },
    2: {
        "id": "II",
        "adjectives": "fair, white, light",
        "reaction": "burns easily, tans minimally"
    },
    3: {
        "id": "III",
        "adjectives": "light brown, medium white, olive tinge",
        "reaction": "burns moderately, tans gradually to light brown"
    },
    4: {
        "id": "IV",
        "adjectives": "moderate brown, olive",
        "reaction": "burns minimally, tans well to moderate brown"
    },
    5: {
        "id": "V",
        "adjectives": "brown, dark brown, deep pigment",
        "reaction": "rarely burns, tans profusely to dark brown"
    },
    6: {
        "id": "VI",
        "adjectives": "dark brown, black, deeply pigmented",
        "reaction": "never burns, deeply pigmented, least sun-sensitive"
    }
}


def get_target_tone(current_fitz, strategy='mixed'):
    """
    Selects a target tone based on training difficulty.
    """
    current = int(current_fitz)
    scale = [1, 2, 3, 4, 5, 6]

    if strategy == 'hard':
        # Diametric Opposite (Maximum Gradient Signal)
        # 1<->6, 2<->5, 3<->4
        return 7 - current

    elif strategy == 'easy':
        # Nearest Neighbor (Subtle Gradient Signal)
        # Randomly pick +1 or -1, staying in bounds
        options = []
        if current > 1: options.append(current - 1)
        if current < 6: options.append(current + 1)
        return random.choice(options)

    elif strategy == 'random':
        # Pure Random (Null Distribution)
        options = [x for x in scale if x != current]
        return random.choice(options)

    elif strategy == 'mixed':
        # Mix of Hard, Easy, and Random
        roll = random.random()
        if roll < 0.4: return get_target_tone(current, 'hard')  # 40% Hard
        if roll < 0.7: return get_target_tone(current, 'easy')  # 30% Easy
        return get_target_tone(current, 'random')  # 30% Random

SKIN_TERMS = [
    "fair", "light", "dark", "brown", "black", "pigmented",
    "erythema", "red", "olive"
]

def skin_tone_changed(orig, cf):
    o = orig.lower()
    c = cf.lower()
    return any(t in o and t not in c for t in SKIN_TERMS) or \
           any(t in c and t not in o for t in SKIN_TERMS)

FORBIDDEN_CAUSAL = [
    "more prone", "less prone", "least prone",
    "higher risk", "lower risk",
    "susceptible", "susceptibility",
    "tendency", "tend to", "predisposed",
    "risk of", "protective"
]

def violates_causal_invariance(orig, cf):
    o, c = orig.lower(), cf.lower()
    for t in FORBIDDEN_CAUSAL:
        if (t in o) != (t in c):
            return True
    return False

def is_noop(orig, cf):
    return orig.strip() == cf.strip()


# -------------------------
# Prompted generation
# -------------------------
def generate_counterfactual(model, tokenizer,
                            TARGET_FITZPATRICK: int,
                            ORIGINAL_NOTE: str) -> str:

    target = FITZPATRICK_PROFILES[TARGET_FITZPATRICK]

    prompt = f"""[INST] You are an expert dermatologist rewriting patient notes for a medical dataset augmentation task.

    **Goal:** Rewrite the **Original Text** to change the patient's skin type to **Fitzpatrick Type {target['id']}**.

    **Target Profile (Type {target['id']}):**
    - **Appearance:** {target['adjectives']}
    - **Sun Reaction:** {target['reaction']}

    **Instructions:**
    1. **Identify** sentences describing skin tone, color, or sun sensitivity.
    2. **Rewrite** those sentences to match the **Target Profile**. 
       - You MUST change descriptive adjectives (e.g., "fair" -> "dark") and visual symptoms (e.g., "bright redness" -> "subtle darkness") to be medically consistent with the new skin tone.
    3.   **If NO skin tone is mentioned:** You must INSERT a brief description of the patient's skin (e.g., "Patient has {target['adjectives']} skin") at the beginning of the note.
    4. **Preserve** all other medical facts and morphology (Lesion size, shape, diagnosis, location) exactly as they are.
    5. **Output** the full, revised medical note. Do not include introductory text.
    
    IMPORTANT CONSTRAINTS (MUST FOLLOW):
    - Do NOT change disease risk, susceptibility, prevalence, or causal explanations.
    - Do NOT change statements about whether a patient is more or less prone to a condition.
    - Skin tone changes must affect ONLY visual appearance, not underlying biology or risk.

    **Original Text:**
    "{ORIGINAL_NOTE}"

    **Rewritten Counterfactual:**
    [/INST]"""

    
    model_input = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=400,
            min_new_tokens=20,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.85,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    description = extract_assistant_text(generated_text)
    description = description.replace("<s>", "").replace("</s>", "").strip()

    if description.startswith('"') and description.endswith('"'):
        description = description[1:-1]

    return description





# --- Example Usage ---
if __name__ == '__main__':
    print("-" * 50)

    embed_model = load_embedding_model()
    model, tokenizer = load_biomistral_model()
    if model is None or tokenizer is None:
        raise RuntimeError("Failed to load model/tokenizer")

    results = load_existing()
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    if not INPUT_JSON.exists():
        print(f"Error: {INPUT_JSON} not found.")
        sys.exit()
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)


    results = {}
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON, "r") as f:
                results = json.load(f)
        except:
            pass

    print(f"Starting processing on {len(data)} items...")

    iterator = data.items() if isinstance(data, dict) else enumerate(data)

    # for key, item in data.values():
    for key, item in iterator:
        if key in results and "completed" in results[key]:
            continue

        if not isinstance(item, dict):
            continue

        img_name = item["Name"]
        original_fitz = int(item["Fitzpatrick"])
        original_text = clean_text(item.get("Description_5", ""))

        if not original_text:
            print(f"skipping{img_name}")
            continue


        print(f"\n--- Generating Counterfactual Description for: ---")
        print(f"Image Name: {img_name}")
        print("-" * 50)

        out_item = {"Name": img_name, "Fitzpatrick": original_fitz, "Original_Text": original_text}

        target_fitz = get_target_tone(original_fitz)
        best_cf = None
        best_score = -1.0
        out_item["cf_status"] = "failed"

        for attempt in range(MAX_RETRIES):

            counter_descr = generate_counterfactual(
                model,
                tokenizer,
                TARGET_FITZPATRICK=target_fitz,
                ORIGINAL_NOTE=original_text
            )

            if not counter_descr:
                continue

            counter_descr = counter_descr.strip()

            # No-op check (early)
            if counter_descr == original_text:
                out_item["cf_status"] = "noop"
                continue

            # Similarity check
            sim_score = get_similarity_score(original_text, counter_descr, embed_model)
            if not (0.85 < sim_score < 0.93):
                continue

            # Causal invariance
            if violates_causal_invariance(original_text, counter_descr):
                continue

            # Skin-tone intervention happened
            if not skin_tone_changed(original_text, counter_descr):
                continue

            # ACCEPT
            best_cf = counter_descr
            best_score = sim_score
            out_item["cf_status"] = "valid"
            break

        if best_cf is None:
            best_cf = original_text

        out_item["Description_5_cf"] = best_cf
        out_item["sim_score"] = best_score


        results[img_name] = out_item

        save_results(results)
        print(f"  Saved {img_name} to {OUT_JSON}")

    print("----Completed Task----")




#