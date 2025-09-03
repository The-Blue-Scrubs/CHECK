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


def create_prompt(data, question_number):
    """
    Creates a prompt for a specific question number
    """
    questions = {
        1: "Definition: Identify the title and purpose of the clinical trial.",
        2: "Condition: Describe the conditions studied.",
        3: "Design Details: Explain how the study was designed, including the number of participants enrolled.",
        4: "Interventions: Describe the interventions investigated in the clinical trial.",
        5: "Study Arms: Explain how the study arms were structured.",
        6: "Eligibility Criteria: Describe eligibility criteria for participation in the study.",
        7: "Primary Outcome: Describe the primary outcome measured.",
        8: "Primary Outcome Statistical Analysis: Describe the statistical methods used to analyze the primary outcome.",
        9: "Primary Outcome Statistical Results: Summarize the statistical results obtained for the primary outcome.",
        10: "Secondary Outcomes Overview: Provide a general summary of the secondary outcomes measured.",
        11: "Statistical Approach: Briefly describe the statistical methods used to analyze the secondary outcomes.",
        12: "Key Results: Highlight the most important statistical results and clinically relevant findings from the secondary outcomes.",
        13: "Serious Adverse Events (SAEs): Summarize the most significant and clinically relevant serious adverse events reported.",
        14: "Non-Serious Adverse Events: Briefly list or group the most frequent non-serious adverse events highlighting those. ",
        15: "Key Observations and Clinical Relevance: Provide a short overview of the overall safety profile based on the adverse events, "
             "focusing on any notable trends or conclusions about tolerability and risk."
    }
    
    prompt = (
        "You are an advanced clinical language model. Your task is to answer the following question about the clinical trial "
        f"{data}. \n\n"
        
        f"QUESTION:\n"
        f"{questions[question_number]}\n\n"
        
        "Use medically precise terminology. Be specific and focused ONLY on answering the requested question. Avoid any introductory or concluding sentences."
    )
    return prompt

def process_model(model_info, model, tokenizer, sampling_params, inference_file_path, input_text_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    SENTINEL_VALUE= np.float32(-100.0)
    keys_to_keep = ['Final_text', 'Conclusion'] ## for input_text --> data for prompt

    # Load the JSON file
    try:
        with open(inference_file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {inference_file_path}: {e}")
        return

    with open(input_text_path, "r", encoding='utf-8') as f:
        full_data = json.load(f)

    # Create input_text as a new dictionary with only desired keys ("Final_text", "Conclusion")
    input_text = {key: full_data[key] for key in keys_to_keep if key in full_data}
        
    total_tasks = len(data)  # Should be 15 items

    with tqdm(total=total_tasks, desc="Processing Reasoning", unit="task") as progress_bar:
        # Process each question in the JSON
        for question_number in range(1, 16):  # Process Q1 to Q15
            key = f"Q{question_number}"
            # Get the text for this question
            inference_text = data[key]
            # print("inference file:", inference_file_path)
            # print("input_text_path:", input_text_path)
            # print("Key:", key)
            # print("inference_text:", inference_text)
            if not inference_text or not isinstance(inference_text, str):
                progress_bar.update(1)
                continue

            # Create your prompt and combine with text
            prompt = create_prompt(input_text, question_number)
            full_text = prompt + "\n\n" + inference_text

            # 2. Tokenize
            all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            prompt_tokens = tokenizer.encode(prompt + "\n\n", add_special_tokens=False)
            prompt_token_count = len(prompt_tokens)

            # 3. Number of tokens we're going to "force"
            num_positions = len(all_tokens) - prompt_token_count
            if num_positions <= 0:
                # If text is too short or missing, skip
                progress_bar.update(1)
                continue

            # 4. Prepare the matrix [num_positions x 102]
            num_features = 102 # 1 gold logprob + 50 top logprobs + 1 gold ID + 50 top IDs
            logprob_token_matrix = np.full((num_positions, num_features), SENTINEL_VALUE, dtype=np.float32)
            # Add a new array to store ranks
            ranks = np.full(num_positions, 100, dtype=np.float32)

            # 5. Forced decoding step-by-step for each token after the prompt
            for pos_idx, i in enumerate(range(prompt_token_count, len(all_tokens))):
                # Context is everything up to (but not including) the next token
                current_context_tokens = all_tokens[:i]
                current_context_text = tokenizer.decode(current_context_tokens, skip_special_tokens=True)
                next_token_id = all_tokens[i]
                # Generate one token, retrieving logprobs
                outputs = model.generate(
                    [current_context_text],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                
                # This dictionary maps {token_id: logprob_entry} (depending on your LLM library)
                token_logprobs = outputs[0].outputs[0].logprobs[0]

                # 5a. GOLD token logprob => column 0, token ID => column 51
                if token_logprobs is not None:
                    if next_token_id in token_logprobs:
                        gold_val = token_logprobs[next_token_id]
                        gold_val_rank = gold_val.rank
                        ranks[pos_idx] = gold_val_rank
                        # Some libraries store a LogProb object with a `.logprob` attribute; 
                        # others store raw floats. Adjust as needed:
                        if hasattr(gold_val, "logprob"):
                            gold_val = gold_val.logprob

                        logprob_token_matrix[pos_idx, 0] = float(gold_val)
                        logprob_token_matrix[pos_idx, 51] = float(next_token_id)
                    else:
                        # The gold token isn't in the returned top tokens
                        logprob_token_matrix[pos_idx, 0] = SENTINEL_VALUE
                        logprob_token_matrix[pos_idx, 51] = float(next_token_id)

                    # 5b. Fill top-50 alternate tokens in columns 1..50 (logprobs) and 52..102 (IDs)
                    sorted_logprobs = sorted(
                        token_logprobs.items(),
                        key=lambda x: float(x[1].logprob if hasattr(x[1], "logprob") else x[1]),
                        reverse=True
                    )
                    
                    # Slice the top-50 tokens right away:
                    sorted_logprobs = sorted_logprobs[:50]

                    # -----------------------------------------------------
                    # 3) Fill columns [1..50] => top-50 token logprobs,
                    #    columns [52..101] => top-50 token IDs.
                    # -----------------------------------------------------
                    for offset, (token_id, entry) in enumerate(sorted_logprobs):
                        prob_val = entry.logprob if hasattr(entry, "logprob") else entry

                        # offset goes from 0..49, so logprob => col (1 + offset), token ID => col (52 + offset)
                        logprob_token_matrix[pos_idx, 1 + offset]  = float(prob_val)
                        logprob_token_matrix[pos_idx, 52 + offset] = float(token_id)

            # 6. Convert the NumPy matrix to a DataFrame for easier CSV output
            logprob_cols = ["gold_lp"] + [f"lp_{c}" for c in range(1, 51)]
            token_id_cols = ["gold_id"] + [f"id_{c}" for c in range(1, 51)]
            all_cols = logprob_cols + token_id_cols

            # Now all_cols has length 1 + 50 + 1 + 50 = 102.
            matrix_df = pd.DataFrame(logprob_token_matrix, columns=all_cols)
            matrix_df['rank'] = ranks

            # 7. Save this matrix to a CSV, one file per row
            output_filename = f"logprob_matrix_{question_number}.csv"
            output_path = os.path.join(output_dir, output_filename)
            matrix_df.to_csv(output_path, index=False)

            progress_bar.update(1)

    print(f"Done! Individual CSVs for each row are saved in '{output_dir}'.")

if __name__ == "__main__":
    # Add proper logging
    logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
# Base paths
    inference_base_path = "Database_dependent_evaluation/Clinical_trials/3-Inference/Model_Answer/Llama3.3-70B/Inference_summary_Paragraph"
    input_text_base_path = "Database_dependent_evaluation/Clinical_trials/3-Inference/Context/Trial_summary"

        models_info = [
        {
            "model_path": "Llama-3.3-70B-Instruct/",  # LLama 3.3-70B 
            "model_id": "model_1"
        }
    #     {
    #         "model_path": "Llama-3.1-70B-Instruct/",  # LLama 3.1-70B
    #         "model_id": "model_2"
    #     }
    #     {
    #         "model_path": "Meta-Llama-3.1-8B-Instruct",  # LLama 3.1-8B
    #         "model_id": "model_3"
    #     }
    #     {
    #         "model_path": "DeepSeek-R1-Distill-Llama-70B",  # DeepSeek-R1-Distill-Llama-70B
    #         "model_id": "model_4" 
    #     }
    #     {
    #         "model_path": "Llama-3.1-Nemotron-70B-Instruct-HF/",  # Llama 3.1-Nemotron-70B
    #         "model_id": "model_6"  
    #     }    
    # ]

    # Get all json files from inference directory
    inference_files = [f for f in os.listdir(inference_base_path) if f.endswith('.json')]
    
    # Process each model
    for model_info in models_info:
        logging.info(f"\nStarting processing with {model_info['model_id']}")
        print(f"\nProcessing with {model_info['model_id']}")
        
        # Check which files have already been processed for this model
        processed_files = set()
        model_output_dir = os.path.join("Paragraph_summary", model_info["model_id"])
        if os.path.exists(model_output_dir):
            processed_files = {
                f for f in os.listdir(model_output_dir) 
                if not f.startswith('.') and
                not f in {'.', '..', '.ipynb_checkpoints'} and
                os.path.isdir(os.path.join(model_output_dir, f))
            }
        
        # Create list of pending files for this model
        all_files = [f for f in inference_files 
                        if f.replace('.json', '') not in processed_files]
        pending_files = sorted(all_files)
        
        print(f"\nDetailed counts for {model_info['model_id']}:")
        print(f"Total files in inference directory: {len(inference_files)}")
        print(f"Already processed files: {len(processed_files)}")
        print(f"Files pending to process: {len(pending_files)}")
        print("\nProcessed directories:", sorted(list(processed_files)))
        
        # Load model and tokenizer
        MODEL_PATH = model_info["model_path"]
        logging.info(f"Loading model from {MODEL_PATH}")
        
        model = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.95,
            max_num_batched_tokens=32768,
            max_logprobs=50,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=1,
            max_tokens=1,
            logprobs=50,
        )

        # Process each pending file for this model
        with tqdm(total=len(pending_files), desc=f"Processing Files - {model_info['model_id']}") as file_progress:
            for inference_file in pending_files:
                try:
                    logging.info(f"Starting to process {inference_file} with {model_info['model_id']}")

                    # Get corresponding input text file
                    input_text_file = inference_file
                    inference_file_path = os.path.join(inference_base_path, inference_file)
                    input_text_path = os.path.join(input_text_base_path, input_text_file)
                    
                    # Check if input text file exists
                    if not os.path.exists(input_text_path):
                        error_msg = f"Warning: No matching input text file found for {inference_file}"
                        print(error_msg)
                        logging.warning(error_msg)
                        file_progress.update(1)
                        continue
                        
                    folder_name = inference_file.replace('.json', '')
                    OUTPUT_DIR = os.path.join("Paragraph_summary", model_info["model_id"], folder_name)
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    
                    print(f"Processing {folder_name} with {model_info['model_id']}...")
                    
                    process_model(model_info, model, tokenizer, sampling_params, 
                                inference_file_path, input_text_path, OUTPUT_DIR)
                    logging.info(f"Successfully processed {inference_file} with {model_info['model_id']}")
                
                    file_progress.update(1)
                except Exception as e:
                    logging.error(f"Error processing {inference_file} with {model_info['model_id']}: {e}")
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

