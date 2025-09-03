import os
import json
import torch
from vllm import LLM, SamplingParams
import time
import glob
from tqdm import tqdm
import re

def clean_deepseek_answer(raw_answer):
    """
    Clean DeepSeek model answer by finding the LAST </think> tag
    by searching backwards from the end
    """
    
    if not raw_answer or not isinstance(raw_answer, str):
        return raw_answer
    
    # Find the LAST occurrence of </think> by searching backwards
    last_think_pos = raw_answer.rfind('</think>')
    
    if last_think_pos != -1:
        # Found </think>, get content after it
        start_pos = last_think_pos + len('</think>')
        content = raw_answer[start_pos:].strip()
        
        if content:  # Only return if there's actual content
            # Basic cleanup: remove any remaining tags and normalize whitespace
            content = re.sub(r'<[^>]+>', '', content)  # Remove any remaining tags
            content = re.sub(r'\s+', ' ', content)     # Normalize whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)  # Normalize paragraph breaks
            return content.strip()
    
    # If no </think> tag found, return empty string
    return ""

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

def process_batch(llm, file_batch, question_number, output_dir, batch_size=8):
    """
    Process a batch of files for a specific question
    """
    prompts = []
    file_paths = []
    max_words = 30000
    
    # Prepare prompts for the batch
    for file_path in file_batch:
        try:
            # Read existing output file if it exists - USE output_dir consistently
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            existing_data = {}
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Skip if this question is already answered
            if f'Q{question_number}' in existing_data:
                continue
                
            with open(file_path, "r", encoding='utf-8') as f:
                full_data = json.load(f)
                original_data = full_data['protocolSection']
                id_module = original_data['identificationModule']
                
                # Extract title text properly (not as JSON)
                if ('officialTitle' in id_module and 
                    id_module['officialTitle'] and 
                    id_module['officialTitle'].strip()):
                    title_text = id_module['officialTitle'].strip()
                else:
                    title_text = id_module['briefTitle'].strip()
            
            # Handle max words limit - truncate the TEXT, not JSON
            words = title_text.split()
            if len(words) > max_words:
                title_text = ' '.join(words[:max_words])
            
            prompt = create_prompt(title_text, question_number)
            prompts.append(prompt)
            file_paths.append(file_path)
            
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            continue
    
    if not prompts:
        return
    
    # Generate text for the batch
    try:
        sampling_params = SamplingParams(temperature=0.3, top_p=0.85, max_tokens=4096)
        outputs = llm.generate(prompts, sampling_params)
        
        # Process and save outputs
        for file_path, output in zip(file_paths, outputs):
            try:
                # Read existing data or create new - USE output_dir consistently
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                existing_data = {}
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                
                # ✨ CLEAN THE ANSWER - Remove DeepSeek reasoning sections
                raw_answer = output.outputs[0].text
                cleaned_answer = clean_deepseek_answer(raw_answer)
                
                # Update with CLEANED answer
                existing_data[f'Q{question_number}'] = cleaned_answer
                
                # Optionally save raw answer for debugging (uncomment if needed)
                # existing_data[f'Q{question_number}_raw'] = raw_answer
                
                # Save updated data
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=4, ensure_ascii=False)
                
                # Print cleaning example for first few files (for verification)
                if len(file_paths) <= 3:  # Show first 3 examples
                    print(f"\n🧹 Cleaning example for {os.path.basename(file_path)}:")
                    print(f"RAW ({len(raw_answer)} chars): {raw_answer[:100]}...")
                    print(f"CLEANED ({len(cleaned_answer)} chars): {cleaned_answer[:100]}...")
                
            except Exception as e:
                print(f"Error saving {file_path}: {str(e)}")
                
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")

def main():
    # Define output directory
    output_dir = "Inference_title_Paragraph"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("🧹 Answer cleaning: ENABLED (extracts ONLY content after last </think>)")
    
    # Initialize model once
    print("Initializing model...")
    llm = LLM("DeepSeek-R1-Distill-Llama-70B/",
              tensor_parallel_size=4,
              gpu_memory_utilization=0.92, 
              max_num_batched_tokens=2048)

    # Get files from input directory
    input_files = sorted(glob.glob("Database_dependent_evaluation/Clinical_trials/3-Inference/Context/Trial_raw_file/*.json"))
    total_input_files = len(input_files)
    print(f"Found {total_input_files} total input files")
    
    if not input_files:
        print("No JSON files found in Original_format directory")
        return

    # Process each question separately
    for question_number in range(1, 16):
        print(f"\nProcessing Question {question_number}/15")
        
        # Get pending files for this question with proper encoding
        pending_files = []
        for f in input_files:
            output_path = os.path.join(output_dir, os.path.basename(f))  # Use output_dir
            try:
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as json_file:
                        existing_data = json.load(json_file)
                        if f'Q{question_number}' not in existing_data:
                            pending_files.append(f)
                else:
                    pending_files.append(f)
            except Exception as e:
                print(f"Error reading {output_path}: {str(e)}")
                pending_files.append(f)
        
        total_files = len(pending_files)
        print(f"Found {total_files} files to process for Q{question_number}")
        
        if total_files == 0:
            continue
        
        # Process in batches - PASS output_dir to process_batch
        batch_size = 100
        for i in range(0, len(pending_files), batch_size):
            batch = pending_files[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            print(f"Processing files {i+1} to {min(i+batch_size, total_files)} of {total_files}")
            process_batch(llm, batch, question_number, output_dir, batch_size)  # Pass output_dir

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds") 