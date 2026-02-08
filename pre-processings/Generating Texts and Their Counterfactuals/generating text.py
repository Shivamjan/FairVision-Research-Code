import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import csv
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# --- Configuration ---
MODEL_ID = "BioMistral/BioMistral-7B"



# load the model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

MAX_RETRIES = 5

def is_valid(text: str) -> bool:
    if not text:
        return False

    bad_endings = [
        "clinical impression:",
        "1. clinical impression:",
        "1. **clinical impression:**"
    ]

    if text.strip().lower().endswith(tuple(bad_endings)):
        return False

    return True

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


def generate_lesion_description(model, tokenizer, fitzpatrick_scale, disease, malignant, temp):  #changes based on metadata

    
    prompt_template_fitz = f"""[INST]
    You are a highly experienced dermatologist specializing in clinical documentation.
    
    Generate a clinical description of a skin lesion based ONLY on the diagnostic and phenotypical information provided below.
    Do NOT include any specific details that are not derivable from these categories, such as:
    - Patient age, gender, or demographics
    - Exact lesion size, dimensions, or measurements
    - Specific anatomical location
    - Duration or history of the lesion
    - Symptoms (pain, itching, bleeding, etc.)
    
    **Input Categories:**
    - Fitzpatrick Scale (Sun-reactive skin type): {fitzpatrick}
    - Nine-Partition Label (Detailed diagnosis category): {nine_partition_label}
    - Three-Partition Label (High-level malignancy status): {three_partition_label}
    
    **Required Output:**
    1. **Clinical Impression:** Based on the diagnosis category, describe the typical morphological features, color characteristics, and border patterns associated with this type of lesion. Reference how the Fitzpatrick skin type may affect the lesion's appearance.
    
    2. **Malignancy Assessment:** State the three-partition classification and what this implies clinically.
    
    3. **Differential Considerations:** List 2-3 alternative diagnoses that share similar features with the nine-partition category provided.
    
    **Important:** Frame your description using general medical terminology (e.g., "lesions of this type typically present with..." rather than "this 2.5cm lesion on the left arm..."). Only describe what can be reasonably inferred from the categorical labels provided.
    [/INST]
    """
    

    prompt_template_ddi_1 = f"""[INST]
    You are a highly experienced dermatologist specializing in clinical documentation.
    
    Generate a clinical description of a skin lesion based ONLY on the diagnostic and phenotypical information provided below.
    Do NOT include any specific details that are not derivable from these categories.

    The disease label is provided **only to guide the general appearance and risk context** and must **not be mentioned, named, or inferable** from the text.

    ---

    **Input Categories:**
    - Fitzpatrick Skin Type: {fitzpatrick}
    - Malignancy Status: {malignant}  (True = malignant, False = benign)
    - Skin Disease Label (conditioning only): {disease}

    ---

    **Writing Guidelines (Important):**
    - Use the disease label **only as a latent conditioning signal** to shape general morphology and risk context.
    - Do NOT mention the disease label, synonyms, or hallmark features that uniquely identify it.
    - Do NOT include patient-specific attributes (age, sex, lesion size, anatomical location, duration).
    - Use general, observational language typical of dermatology clinical notes.
    - Frame statements as tendencies (e.g., “Lesions of this type typically…”).

    ---

    **Required Output Format:**

    1. **Clinical Impression:**  
    Describe the lesion’s *typical morphological appearance* in general terms, including features such as symmetry or asymmetry, border definition, surface texture, and pigmentation patterns.  
    Explicitly describe how these features may appear on the specified Fitzpatrick skin type (e.g., pigmentation contrast, visibility of erythema).

    ---
    Begin the response directly with section 1, using professional clinical documentation style.
    [/INST]
    """
    

    
    model_input = tokenizer(
        prompt_template_ddi_1,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=400,
            min_new_tokens=80,
            do_sample=True,
            temperature=temp,
            top_p=0.85,
            pad_token_id=tokenizer.eos_token_id
        )


    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    #look for the last '[/INST]' and start from there. worked for fitz
    # if "[/INST]" in generated_text:
    #     description_start = generated_text.rfind("[/INST]") + len("[/INST]")
    #     return generated_text[description_start:].strip()
    # return generated_text.strip()

    #for ddi
    if "[/INST]" in generated_text:
        description = generated_text.split("[/INST]", 1)[-1].strip()
    else:
        description = generated_text

    description = description.replace("</s>", "").replace("<s>", "").strip()

    return description


CSV_PATH = Path("path/to/your/metadata_csv/")
OUT_JSON = Path("output/path/to/store/json/")


def save_everything(data: Dict[str, Any]) -> None:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


# --- Example Usage ---
if __name__ == '__main__':
    print("-" * 50)

    results = load_existing()
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    model, tokenizer = load_biomistral_model()

    if model is None or tokenizer is None:
        raise RuntimeError("Failed to load model/tokenizer")

    for idx, row in df.iterrows():
        # print(idx, row)
        img_name = str(row['hasher'])
        if img_name in results:
            print(f"[{idx}] Skipping {img_name} (already processed).")
            continue


        fitzpatrick = row['fitzpatrick']
        disease = str(row['disease'])
        malignant = bool(row['malignant'])

        print(f"\n--- Generating Description for: ---")
        print(f"Image Name: {img_name}")
        print("-" * 50)

        item = {"Name": img_name, "Fitzpatrick": fitzpatrick, "Malignant": malignant, "disease": disease}

        for temp in np.arange(0.1, 0.6, 0.1):
            #print(f"for the {temp} temperature:")
            for attempt in range(MAX_RETRIES):
                descr = generate_lesion_description(model,
                    tokenizer,
                    fitzpatrick,
                    disease,
                    malignant,
                    float(temp))

                if descr and is_valid(descr):
                    break
            else:
                descr = None  # mark as failed

            key = f"Description_{int(temp * 10)}"  # 1,2,3,4,5
            item[key] = descr

        results[img_name] = item

        save_everything(results)
        print(f"  Saved {img_name} to {OUT_JSON}")

    print("----Completed Task----")



# 
