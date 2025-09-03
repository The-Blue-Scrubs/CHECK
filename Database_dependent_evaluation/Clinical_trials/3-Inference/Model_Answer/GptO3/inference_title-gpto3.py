import os
import json
import time
import glob
from tqdm import tqdm
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_trial_gpt_5.log'),
        logging.StreamHandler()
    ]
)

# Initialize OpenAI client
client = OpenAI(api_key='')

def get_completion(prompt):
    """Get completion from OpenAI GPT-5 API"""
    try:
        response = client.chat.completions.create(
            model="o3-2025-04-16",  # GPT-o3 model
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        return None

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

def process_batch(file_batch, question_number, output_dir):
    """
    Process a batch of files for a specific question using OpenAI API
    """
    max_words = 30000
    
    # Process each file individually (API calls are sequential)
    for file_path in tqdm(file_batch, desc=f"Processing Q{question_number} files"):
        try:
            # Read existing output file if it exists
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
            
            # Create prompt and get completion
            prompt = create_prompt(title_text, question_number)
            answer = get_completion(prompt)
            
            if answer:
                # Update with answer (no cleaning needed for GPT-o3)
                existing_data[f'Q{question_number}'] = answer
                
                # Save updated data
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=4, ensure_ascii=False)
                
                logging.info(f"Processed {os.path.basename(file_path)} - Q{question_number}")
            else:
                logging.warning(f"No response from API for {os.path.basename(file_path)} - Q{question_number}")
            
            # Add delay to respect rate limits
            time.sleep(1)
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            continue

def main():
    # Define output directory
    output_dir = "Inference_title_Paragraph"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print("🤖 Clinical Trial Analysis - GPT-o3 Mode")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print("🤖 Model: GPT-5 (OpenAI API)")
    print("🎯 Processing: Clinical trial questions 1-15")
    print("🧹 Answer cleaning: NOT NEEDED (GPT-o3 doesn't use <think> tags)")
    print("=" * 60)

    # Get files from input directory
    input_files = sorted(glob.glob("Database_dependent_evaluation/Clinical_trials/3-Inference/Context/Trial_raw_file/*.json"))
    total_input_files = len(input_files)
    logging.info(f"Found {total_input_files} total input files")
    
    if not input_files:
        logging.warning("No JSON files found in Original_format directory")
        return

    # Process each question separately
    for question_number in range(1, 16):
        logging.info(f"Processing Question {question_number}/15")
        
        # Get pending files for this question
        pending_files = []
        for f in input_files:
            output_path = os.path.join(output_dir, os.path.basename(f))
            try:
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as json_file:
                        existing_data = json.load(json_file)
                        if f'Q{question_number}' not in existing_data:
                            pending_files.append(f)
                else:
                    pending_files.append(f)
            except Exception as e:
                logging.error(f"Error reading {output_path}: {str(e)}")
                pending_files.append(f)
        
        total_files = len(pending_files)
        logging.info(f"Found {total_files} files to process for Q{question_number}")
        
        if total_files == 0:
            continue
        
        # Process files (no batching needed for API calls)
        process_batch(pending_files, question_number, output_dir)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"\n✅ Processing complete! Total time: {end_time - start_time:.2f} seconds") 
