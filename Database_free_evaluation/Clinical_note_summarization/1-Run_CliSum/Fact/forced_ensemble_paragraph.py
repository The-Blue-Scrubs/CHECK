import os
import json
import math
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gc


def create_prompt(user_question):
    """
    Creates a chat-formatted prompt for Llama 3.3
    
    Args:
        user_question (str): The question from the user
    
    Returns:
        str: Formatted chat prompt ready for Llama 3.3
    """
    
    system_message = "You are an expert medical professional. Summarize the radiology report findings into an impression with minimal text. "

    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{system_message + user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""    
    
    return prompt 


def process_model(model_info, model, tokenizer, sampling_params, input_json_path, output_dir):
    """
    Modified to work with the new NCT*.json format
    """
    os.makedirs(output_dir, exist_ok=True)
    SENTINEL_VALUE = np.float32(-10.0)

    # Load the JSON file
    try:
        with open(input_json_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_json_path}: {e}")
        return

    # Extract the prompt and completion from the new format
    input_text_question = data['inputs']
    inference_text = data['target']               # target - FACT /  target_hallucination - HALLUCINATION
        
    with tqdm(total=1, desc="Processing Example", unit="task") as progress_bar:
        # Always use Q1 for consistency
        if not inference_text or not isinstance(inference_text, str):
            progress_bar.update(1)
            return

        # Use the formatted prompt directly
        prompt = create_prompt(input_text_question)
        full_text = prompt + inference_text

        # Tokenize
        all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_count = len(prompt_tokens)

        num_positions = len(all_tokens) - prompt_token_count
        if num_positions <= 0:
            progress_bar.update(1)
            return

        # Matrix creation
        num_features = 102  # 1 gold logprob + 50 top logprobs + 1 gold ID + 50 top IDs
        logprob_token_matrix = np.full((num_positions, num_features), SENTINEL_VALUE, dtype=np.float32)
        ranks = np.full(num_positions, 100, dtype=np.float32)

        # Forced decoding step-by-step
        for pos_idx, i in enumerate(range(prompt_token_count, len(all_tokens))):
            current_context_tokens = all_tokens[:i]
            current_context_text = tokenizer.decode(current_context_tokens, skip_special_tokens=False)
            next_token_id = all_tokens[i]
            
            outputs = model.generate(
                [current_context_text],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            token_logprobs = outputs[0].outputs[0].logprobs[0]

            # print("---------------------------------------------------------")
            # print("Prompt:", current_context_text)
            # print("---    -------     ------     -------     ---------------")
            # print("Inf_token:", next_token_id, "Inf_text:", tokenizer.decode([next_token_id], skip_special_tokens=False))
            # print("Log=", token_logprobs)
            # print("---------------------------------------------------------")

            if token_logprobs is not None:
                if next_token_id in token_logprobs:
                    gold_val = token_logprobs[next_token_id]
                    gold_val_rank = gold_val.rank
                    ranks[pos_idx] = gold_val_rank
                    
                    if hasattr(gold_val, "logprob"):
                        gold_val = gold_val.logprob

                    logprob_token_matrix[pos_idx, 0] = float(gold_val)
                    logprob_token_matrix[pos_idx, 51] = float(next_token_id)
                else:
                    logprob_token_matrix[pos_idx, 0] = SENTINEL_VALUE
                    logprob_token_matrix[pos_idx, 51] = float(next_token_id)

                sorted_logprobs = sorted(
                    token_logprobs.items(),
                    key=lambda x: float(x[1].logprob if hasattr(x[1], "logprob") else x[1]),
                    reverse=True
                )
                
                sorted_logprobs = sorted_logprobs[:50]

                for offset, (token_id, entry) in enumerate(sorted_logprobs):
                    prob_val = entry.logprob if hasattr(entry, "logprob") else entry
                    logprob_token_matrix[pos_idx, 1 + offset] = float(prob_val)
                    logprob_token_matrix[pos_idx, 52 + offset] = float(token_id)

        # Create DataFrame and save
        logprob_cols = ["gold_lp"] + [f"lp_{c}" for c in range(1, 51)]
        token_id_cols = ["gold_id"] + [f"id_{c}" for c in range(1, 51)]
        all_cols = logprob_cols + token_id_cols

        matrix_df = pd.DataFrame(logprob_token_matrix, columns=all_cols)
        matrix_df['rank'] = ranks

        output_filename = f"logprob_matrix_1.csv"  # Always use 1 for consistency
        output_path = os.path.join(output_dir, output_filename)
        matrix_df.to_csv(output_path, index=False)

        progress_bar.update(1)

    print(f"Done! CSV saved in '{output_dir}'.")


if __name__ == "__main__":
    # Add proper logging
    logging.basicConfig(
        filename='processing.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Base paths
    input_base_path = "Database_free_evaluation/Clinical_note_summarization/Data"  # Directory containing NCT*.json files
    granularity = "Paragraph_fact"

        models_info = [
        # {
        #     "model_path": "Llama-3.3-70B-Instruct/",  # LLama 3.3-70B 
        #     "model_id": "model_1"
        # },
        # {
        #     "model_path": "Llama-3.1-70B-Instruct/",  # LLama 3.1-70B
        #     "model_id": "model_2"
        # },
        {
            "model_path": "Meta-Llama-3.1-8B-Instruct",  # LLama 3.1-8B
            "model_id": "model_3"
        }
        # {
        #     "model_path": "DeepSeek-R1-Distill-Llama-70B",  # DeepSeek-R1-Distill-Llama-70B
        #     "model_id": "model_4" 
        # },
        # {
        #     "model_path": "Llama-3.1-Nemotron-70B-Instruct-HF/",  # Llama 3.1-Nemotron-70B
        #     "model_id": "model_6"  
        # }    
    ]

    # Get all NCT json files
    input_files = [f for f in os.listdir(input_base_path) if f.startswith('NCT') and f.endswith('.json')]
    
    # Process each model
    for model_info in models_info:
        logging.info(f"\nStarting processing with {model_info['model_id']}")
        print(f"\nProcessing with {model_info['model_id']}")
        
        # Check processed files
        processed_files = set()
        model_output_dir = os.path.join(granularity, model_info["model_id"])
        if os.path.exists(model_output_dir):
            processed_files = {
                f for f in os.listdir(model_output_dir) 
                if not f.startswith('.') and
                not f in {'.', '..', '.ipynb_checkpoints'} and
                os.path.isdir(os.path.join(model_output_dir, f))
            }
        
        # Create list of pending files
        pending_files = sorted([f for f in input_files 
                              if f.replace('.json', '') not in processed_files])
        
        print(f"\nDetailed counts for {model_info['model_id']}:")
        print(f"Total NCT files: {len(input_files)}")
        print(f"Already processed files: {len(processed_files)}")
        print(f"Files pending to process: {len(pending_files)}")
        
        # Load model and tokenizer
        MODEL_PATH = model_info["model_path"]
        logging.info(f"Loading model from {MODEL_PATH}")
        
        model = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.95,
            max_num_batched_tokens=2048,
            max_logprobs=50,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=1,
            max_tokens=1,
            logprobs=50,
        )

        # Process each NCT file
        with tqdm(total=len(pending_files), desc=f"Processing Files - {model_info['model_id']}") as file_progress:
            for input_file in pending_files:
                try:
                    logging.info(f"Starting to process {input_file} with {model_info['model_id']}")

                    input_file_path = os.path.join(input_base_path, input_file)
                    
                    folder_name = input_file.replace('.json', '')
                    OUTPUT_DIR = os.path.join(granularity, model_info["model_id"], folder_name)
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    
                    print(f"Processing {folder_name} with {model_info['model_id']}...")
                    
                    process_model(model_info, model, tokenizer, sampling_params, 
                                input_file_path, OUTPUT_DIR)
                    logging.info(f"Successfully processed {input_file} with {model_info['model_id']}")
                
                    file_progress.update(1)
                except Exception as e:
                    logging.error(f"Error processing {input_file} with {model_info['model_id']}: {e}")
                    file_progress.update(1)
                    continue
                finally:
                    gc.collect()
        
        # Clean up model before loading next one
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"Completed processing with {model_info['model_id']}")