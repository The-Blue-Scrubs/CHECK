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
MODEL_NAME = "gpt-5-2025-08-07"
MAX_TOKENS = 128


def create_false_statement_prompt(findings, impression):
    """
    Creates a prompt to generate a clinically plausible but incorrect (hallucinated) IMPRESSION
    from the given radiology FINDINGS and TRUE_IMPRESSION. General style (no strict length rules).
    """
    system_message = (
        "You are an expert radiologist generating intentionally incorrect (hallucinated) IMPRESSIONS for model evaluation ONLY.\n\n"
        "GOAL:\n"
        "• Produce an impression that reads like a standard radiology impression but is medically WRONG relative to the FINDINGS.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "• Prefer a single concise sentence (two max), professional radiology style.\n"
        "• Summarize imaging only—no recommendations, management, or new sections.\n"
        "• Use standard radiology terminology; accepted synonyms are fine.\n\n"
        "HOW TO MAKE IT WRONG — apply EXACTLY ONE of the following changes:\n"
        "• NEGATION_FLIP (present ↔ absent) — e.g., 'no effusion' → 'small effusion'.\n"
        "• LOCATION_SWAP (right ↔ left; upper ↔ lower; lobar/hemithorax swaps consistent with anatomy).\n"
        "• SEVERITY_SHIFT (mild ↔ moderate/severe).\n"
        "• ENTITY_SWAP between commonly confused entities (atelectasis ↔ consolidation; edema ↔ effusion).\n"
        "• TEMPORAL_STATUS (stable ↔ new/worsened) if prior status is mentioned.\n"
        "• QUANT_TWEAK (realistic size/extent change when numbers exist).\n\n"
        "CONSTRAINTS:\n"
        "• Base terminology on the FINDINGS or clinically accepted synonyms.\n"
        "• Make exactly one factual error that a clinician could detect by comparing to the FINDINGS.\n"
        "• Avoid internal contradictions; do not invent devices, prior exams, or history not in FINDINGS.\n"
        "• No non-medical content.\n\n"
        "RETURN:\n"
        "• Return ONLY the hallucinated impression text."
    )

    user_message = (
        "Rewrite the TRUE_IMPRESSION into a hallucinated version that follows the rules above.\n\n"
        f"FINDINGS:\n{findings}\n\n"
        f"TRUE_IMPRESSION:\n{impression}\n\n"
        "Instructions:\n"
        "1) Keep it impression-like and concise.\n"
        "2) Apply EXACTLY ONE allowed error type so the result contradicts the FINDINGS.\n"
        "3) Maintain professional tone; use standard radiology terms or synonyms.\n"
        "Generate the hallucinated IMPRESSION now:"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def process_json_files(input_dir, output_dir):
    """Process all JSONL files and save each example as individual JSON file"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of JSONL files
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    logging.info(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Track processing statistics
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    # Process each file
    for jsonl_file in tqdm(sorted(jsonl_files), desc="Processing files"):
        input_path = os.path.join(input_dir, jsonl_file)
        
        try:
            # Read input JSONL (line by line)
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error in {jsonl_file} line {line_num}: {str(e)}")
                        total_failed += 1
                        continue
            
                    # Get the idx for filename
                    idx = data.get('idx', f"{jsonl_file}_{line_num}")  # Fallback if no idx
                    output_filename = f"{idx}.json"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Skip if already processed
                    if os.path.exists(output_path):
                        logging.info(f"Skipping idx {idx} - already processed")
                        total_skipped += 1
                        continue
                    
                    # Get the original text
                    findings = data.get('inputs', '')
                    impression = data.get('target', '')
                    
                    
                    if impression:
                        try:
                            # Create messages for API call
                            messages = create_false_statement_prompt(findings, impression)
                            
                            # Make API call
                            completion = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages
                            )
                            
                            # Get the hallucinated version
                            hallucinated_text = completion.choices[0].message.content.strip()
                            
                            # Create output data structure
                            output_data = data.copy()  # Keep all original data
                            output_data['target_hallucination'] = hallucinated_text  # Add hallucinated version as new key
                            
                            # Save individual JSON file
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(output_data, f, indent=2, ensure_ascii=False)
                            
                            logging.info(f"Successfully processed and saved {idx}.json")
                            total_processed += 1
                            
                        except Exception as e:
                            logging.error(f"Error in API call for {jsonl_file} line {line_num} (idx {idx}): {str(e)}")
                            total_failed += 1
                            continue
                        
                        # Add delay to respect API rate limits
                        time.sleep(0.5)
                    else:
                        logging.warning(f"Missing impression for {jsonl_file} line {line_num} (idx {idx})")
                        total_failed += 1
            
        except Exception as e:
            logging.error(f"Error processing file {jsonl_file}: {str(e)}")
            continue
    
    # Log final statistics
    logging.info(f"\n📊 PROCESSING SUMMARY:")
    logging.info(f"Total processed: {total_processed}")
    logging.info(f"Total skipped (already done): {total_skipped}")
    logging.info(f"Total failed: {total_failed}")
    logging.info(f"Output directory: {output_dir}")

def main():
    input_dir = "Original_data/" # Original Database
    output_dir = "Data/"  # Directory for individual JSON files
    
    start_time = time.time()
    process_json_files(input_dir, output_dir)
    end_time = time.time()
    
    logging.info(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
