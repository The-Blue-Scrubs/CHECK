import tiktoken
from openai import OpenAI
import numpy as np
import pandas as pd
import json
import os
from time import sleep
from tqdm import tqdm

# Initialize OpenAI client and tokenizer
client = OpenAI()
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def process_conversation(json_path, output_dir, prompt_idx):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the NCT ID from the filename
    nct_id = os.path.basename(json_path).replace('.json', '')
    
    # Create specific output directory for this NCT
    nct_output_dir = os.path.join(output_dir, nct_id)
    os.makedirs(nct_output_dir, exist_ok=True)
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    SYSTEM_PROMPT = (
    "You are a clinical expert. Evaluate the assistant's response for any medical hallucinations "
    "(incorrect or fabricated information). Respond only with 'Yes' if hallucinations are present "
    "or 'No' if the assistant's response is factually accurate."
    )
    
    
    user_content = data['prompts'][prompt_idx]
    target_completion = data['target_completion_hallucination']
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_completion}
            ],
            temperature=0.0,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=20
        )

        # print("------------------------------")
        # print(f"Processing {nct_id}")
        # print(completion.choices[0].message)
        # print(completion.choices[0].logprobs)
        # print("------------------------------")

        # Rest of the processing remains the same
        token_logprobs = completion.choices[0].logprobs.content
        num_tokens = len(token_logprobs)
        
        logprob_matrix = np.full((num_tokens, 42), -100.0, dtype=np.float32)
        ranks = np.full(num_tokens, 100, dtype=np.int32)
        
        for pos_idx, token_info in enumerate(token_logprobs):
            gold_token = token_info.token
            gold_logprob = token_info.logprob
            
            gold_token_ids = tokenizer.encode(gold_token)
            gold_token_id = gold_token_ids[0] if gold_token_ids else -1
            
            logprob_matrix[pos_idx, 0] = gold_logprob
            logprob_matrix[pos_idx, 21] = gold_token_id
            
            sorted_logprobs = sorted(token_info.top_logprobs, 
                                   key=lambda x: x.logprob, 
                                   reverse=True)
            
            rank = 1
            for top_token in sorted_logprobs:
                if top_token.token == gold_token:
                    ranks[pos_idx] = rank
                    break
                rank += 1
            
            for idx, top_token in enumerate(sorted_logprobs[:20]):
                token_ids = tokenizer.encode(top_token.token)
                token_id = token_ids[0] if token_ids else -1
                
                logprob_matrix[pos_idx, 1 + idx] = top_token.logprob
                logprob_matrix[pos_idx, 22 + idx] = token_id
        
        logprob_cols = ["gold_lp"] + [f"lp_{i}" for i in range(1, 21)]
        token_id_cols = ["gold_id"] + [f"id_{i}" for i in range(1, 21)]
        all_cols = logprob_cols + token_id_cols
        
        matrix_df = pd.DataFrame(logprob_matrix, columns=all_cols)
        matrix_df['rank'] = ranks
        matrix_df['gold_token_text'] = [tokenizer.decode([int(tid)]) 
                                      for tid in matrix_df['gold_id']]
        
        # Adjust output filename based on option_idx
        output_filename = f"logprob_matrix_1.csv"
        output_path = os.path.join(nct_output_dir, output_filename)
        matrix_df.to_csv(output_path, index=False)
        
        print(f"Processed {len(token_logprobs)} tokens for {nct_id}. Results saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing file {nct_id}: {e}")
        return False

def process_all_files(input_folder, output_dir, prompt_idx):
    # Get all JSON files in the input folder
    json_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file with progress bar
    successful = 0
    failed = 0
    
    with tqdm(total=len(json_files), desc="Processing files") as pbar:
        for json_file in json_files:
            json_path = os.path.join(input_folder, json_file)
            nct_id = json_file.replace('.json', '')
            
            # Check if this specific output already exists
            expected_output = f"logprob_matrix_1.csv"
            if os.path.exists(os.path.join(output_dir, nct_id, expected_output)):
                print(f"Skipping {nct_id} - {expected_output} already processed")
                pbar.update(1)
                continue
                
            # Process the file with current option
            success = process_conversation(json_path, output_dir, prompt_idx)
            if success:
                successful += 1
            else:
                failed += 1
                
            pbar.update(1)
            sleep(0.5)  # Rate limiting
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} runs")
    print(f"Failed: {failed} runs")
    print(f"Total files processed: {len(json_files)}")

if __name__ == "__main__":
    input_folder = "Database_free_evaluation/HealthBench/Data/data"
    

    # Specify which prompt index to use (0-4)
    prompt_idx = 0  # Change this for different runs
    output_dir = "Paragraph_hallucination/model_1"
    
    process_all_files(input_folder, output_dir, prompt_idx)