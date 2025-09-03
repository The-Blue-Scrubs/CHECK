import json
import os
from openai import OpenAI
from tqdm import tqdm
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('atomic_claims_processing.log'),
        logging.StreamHandler()
    ]
)

# Initialize OpenAI client
client = OpenAI(api_key='')

# Configuration
MODEL_NAME = "gpt-4o-2024-08-06"
MAX_TOKENS = 1024


def create_false_statement_prompt(original_text):
    """Creates a prompt to generate a false version of the original text with controlled modifications"""
    
    system_message = (
        "You are an expert in medical content tasked with creating false medical statements that maintain "
        "professional credibility while containing verifiable inaccuracies. Follow these precise guidelines:\n\n"
        
        "REQUIRED MODIFICATIONS:\n"
        "1. Change numerical values (doses, percentages, frequencies) to incorrect but plausible numbers\n"
        "2. Reverse or modify causal relationships between symptoms and conditions\n"
        "3. Substitute correct treatments with incorrect but related medical interventions\n"
        "4. Alter the timing or progression of medical conditions\n"
        "5. Modify anatomical or physiological mechanisms while keeping them plausible\n\n"
        
        "STRICT CONSTRAINTS:\n"
        "1. Maintain all medical terminology - only change their relationships or applications\n"
        "2. Keep the same paragraph structure, length, and writing style\n"
        "3. Ensure all modifications are medically relevant (no non-medical content)\n"
        "4. Preserve the professional tone and grammatical accuracy\n"
        "5. Make errors clear enough for medical professionals to identify\n\n"
        
        "PROHIBITED:\n"
        "1. Do not introduce non-medical or absurd content\n"
        "2. Do not change the basic medical context or condition being discussed\n"
        "3. Do not add new sections or significantly alter the text structure\n"
        "4. Do not use colloquial language or unprofessional terms\n"
    )
    
    user_message = (
        f"Create a false version of this medical text while following the above guidelines exactly:\n\n"
        f"ORIGINAL TEXT:\n{original_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Keep the same structure and format\n"
        "2. Maintain similar length and complexity\n"
        "3. Make specific, verifiable changes to medical facts\n"
        "4. Ensure all modifications are professionally plausible\n"
        "Generate the false version now:"
    )
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def process_json_files(input_dir, output_dir):
    """Process all JSON files in the input directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for json_file in tqdm(sorted(json_files), desc="Processing files"):
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        
        # Skip if already processed
        if os.path.exists(output_path):
            logging.info(f"Skipping {json_file} - already processed")
            continue
        
        try:
            # Read input JSON
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get the original text
            target_completion = data.get('target_completion', '')
            
            if target_completion:
                try:
                    # Create messages for API call
                    messages = create_false_statement_prompt(target_completion)
                    
                    # Make API call
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1024,  # Adjust based on your needs
                        top_p=0.9,
                    )
                    
                    # Get the hallucinated version
                    hallucinated_text = completion.choices[0].message.content.strip()
                    
                    # Create output data structure
                    output_data = data.copy()  # Keep all original data
                    output_data['target_completion_hallucination'] = hallucinated_text  # Add hallucinated version as new key
                    
                    # Save to output file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                    
                    logging.info(f"Successfully processed {json_file}")
                    
                except Exception as e:
                    logging.error(f"Error in API call for {json_file}: {str(e)}")
                    continue
                
                # Add delay to respect API rate limits
                time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error processing file {json_file}: {str(e)}")
            continue

def main():
    input_dir = "HealthBench"
    output_dir = "Database_free_evaluation/HealthBench/Data/data"
    
    start_time = time.time()
    process_json_files(input_dir, output_dir)
    end_time = time.time()
    
    logging.info(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()