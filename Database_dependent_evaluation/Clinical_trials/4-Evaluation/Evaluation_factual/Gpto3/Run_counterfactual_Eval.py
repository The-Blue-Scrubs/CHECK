import os
import logging
import json
import time
from tqdm import tqdm
import openai
from openai import OpenAI

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

# @backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_completion(system_message, user_message):
    """Get completion from OpenAI API with exponential backoff retry"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",  # or your preferred model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=750,
            top_p=0.85
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return None

def create_prompt_counter_factuality_evaluation(context, sentence):
    """Creates a prompt for counterfactual evaluation"""
    system_message = """You are an advanced clinical language model. Your task is to answer whether a statement is contradicted by a given context.
    
Important: Always end your explanation with your final answer in capital letters (YES or NO) on a new line."""
    
    user_message = (
        "The **statement**.\n\n"
        "STATEMENT:"
        f"{sentence}\n\n"
        
        "The **context**.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        
        "Is the statement contradicted by the context above?\n\n"
        "Please explain your reasoning, and end with your answer (YES or NO) on a new line."
    )
    
    return system_message, user_message

def get_last_word(text):
    """Extract the last word from the text"""
    if text:
        # Split by spaces and get last non-empty word
        words = [word for word in text.split() if word]
        if words:
            return words[-1].lower().strip('.,!?')
    return None

def get_all_json_files(statement_folder, context_folder):
    """Get all JSON files that exist in both statement and context folders"""
    try:
        # Get all JSON files from statement folder
        statement_files = set(f for f in os.listdir(statement_folder) if f.endswith('.json'))
        
        # Get all JSON files from context folder  
        context_files = set(f for f in os.listdir(context_folder) if f.endswith('.json'))
        
        # Return intersection - files that exist in both folders
        available_files = list(statement_files.intersection(context_files))
        available_files.sort()  # Sort for consistent processing order
        
        logging.info(f"Found {len(statement_files)} files in statement folder")
        logging.info(f"Found {len(context_files)} files in context folder")
        logging.info(f"Found {len(available_files)} files available for processing")
        
        return available_files
        
    except Exception as e:
        logging.error(f"Error scanning directories: {str(e)}")
        return []

def process_all_files(statement_folder, context_folder, output_folder):
    """Process all available files using OpenAI API for counterfactual analysis"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all available files from both folders
        all_files = get_all_json_files(statement_folder, context_folder)
        
        if not all_files:
            logging.warning("No files found to process")
            return
        
        # Filter out already processed files
        processed_files = set(f for f in os.listdir(output_folder) if f.endswith('.json'))
        pending_files = [f for f in all_files if f not in processed_files]
        
        logging.info(f"Total files available: {len(all_files)}")
        logging.info(f"Already processed: {len(processed_files)}")
        logging.info(f"Files to process: {len(pending_files)}")
        
        if not pending_files:
            logging.info("All files have already been processed!")
            return
        
        # Process files with progress bar
        for filename in tqdm(pending_files, desc="Processing files"):
            try:
                # Construct file paths
                statement_path = os.path.join(statement_folder, filename)
                context_path = os.path.join(context_folder, filename)
                
                # Verify both files exist
                if not os.path.exists(statement_path):
                    logging.warning(f"Statement file not found: {filename}")
                    continue
                    
                if not os.path.exists(context_path):
                    logging.warning(f"Context file not found: {filename}")
                    continue
                
                # Load files
                with open(statement_path, "r", encoding='utf-8') as f:
                    statement_data = json.load(f)
                with open(context_path, "r", encoding='utf-8') as f:
                    context_data = json.load(f)
                
                # Initialize results
                file_results = {}
                questions_processed = 0
                
                # Process each question
                for q_num in range(1, 16):
                    key = f"Q{q_num}"
                    if key in statement_data:
                        sentence = statement_data[key]
                        context = context_data.get("Final_text", "")
                        
                        if not context:
                            logging.warning(f"No 'Final_text' found in context for {filename}")
                            continue
                        
                        system_message, user_message = create_prompt_counter_factuality_evaluation(context, sentence)
                        
                        # Get completion from API
                        evaluation_text = get_completion(system_message, user_message)
                        
                        if evaluation_text:
                            # Get the last word as label
                            label = get_last_word(evaluation_text)
                            
                            file_results[key] = {
                                'statement': statement_data[key],
                                'evaluation': evaluation_text,
                                'label': label if label in ['yes', 'no'] else 'unknown'
                            }
                            questions_processed += 1
                        else:
                            logging.warning(f"No response from API for {filename} {key}")
                            file_results[key] = {
                                'statement': statement_data[key],
                                'error': 'No API response',
                                'label': 'unknown'
                            }
                
                # Save results after each file is processed
                output_path = os.path.join(output_folder, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(file_results, f, indent=2, ensure_ascii=False)
                
                logging.info(f"Processed {filename}: {questions_processed} questions")
                
                # Add a small delay between files to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                continue
        
        logging.info("Counterfactual analysis processing complete!")
        
        # Print final statistics
        final_processed = len([f for f in os.listdir(output_folder) if f.endswith('.json')])
        logging.info(f"Final statistics: {final_processed}/{len(all_files)} files processed")
        
    except Exception as e:
        logging.error(f"Error initializing process: {str(e)}")

if __name__ == "__main__":
    try:
        # Define folders
        statement_folder = "Database_dependent_evaluation/Clinical_trials/3-Inference/Model_Answer/GptO3/Inference_title_Paragraph"
        context_folder = "Database_dependent_evaluation/Clinical_trials/3-Inference/Context/Trial_summary"
        output_folder = "Paragraph_level"

        # Print configuration
        print("🔄 Counterfactual Analysis Processing - All Files Mode")
        print("=" * 60)
        print(f"📁 Statement folder: {statement_folder}")
        print(f"📁 Context folder: {context_folder}")
        print(f"📁 Output folder: {output_folder}")
        print("🤖 Model: gpt-4o-2024-08-06")
        print("🎯 Processing: ALL available files")
        print("🔍 Analysis type: COUNTERFACTUAL (contradiction detection)")
        print("=" * 60)
        
        # Process all available files
        process_all_files(
            statement_folder=statement_folder,
            context_folder=context_folder,
            output_folder=output_folder
        )
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(f"Error in main: {str(e)}") 