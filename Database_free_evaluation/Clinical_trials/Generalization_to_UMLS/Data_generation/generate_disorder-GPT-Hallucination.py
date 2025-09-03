import json
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
MAX_TOKENS = 350
TEMPERATURE = 0.3

def create_false_statement_prompt(original_text, question_type):
    """Creates a prompt to generate a false version of the original text"""
    
    system_message = (
        "You are an expert in medical content who has been tasked with creating false medical statements "
        "that mimic the style and structure of real medical text. Your task is to create a false version "
        "of a given medical statement that:\n"
        "1. Maintains the same medical terminology and professional tone\n"
        "2. Follows the same paragraph structure and length\n"
        "3. Contains clear factual errors that a medical professional would recognize\n"
        "4. Includes incorrect relationships between medical concepts\n"
        "5. Presents plausible but incorrect mechanisms or explanations\n"
        "6. Avoids obviously absurd or non-medical content\n"
        "7. Maintains grammatical correctness and professional writing style"
    )
    
    user_message = (
        f"Create a false version of this medical text about {question_type}. "
        "Maintain the same structure and style but make the content factually incorrect "
        "in a way that seems plausible but is clearly false to medical experts:\n\n"
        f"ORIGINAL TEXT:\n{original_text}"
    )
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def generate_false_statements(input_file, output_file):
    """Read the original answers and generate false versions"""
    
    try:
        # Read the original file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        question_types = {
            'Q1': 'Definition and Pathophysiology',
            'Q2': 'Risk Factors',
            'Q3': 'Symptoms',
            'Q4': 'Tests',
            'Q5': 'Differentiation',
            'Q6': 'First-Line Treatment',
            'Q7': 'Complications'
        }
        
        # Process each sample
        false_data = []
        for sample in tqdm(data, desc="Processing samples"):
            false_sample = {
                'index': sample['index'],
                'AUI': sample['AUI'],
                'CUI': sample['CUI'],
                'term': sample['term'],
                'Definition': sample['Definition'],
                'reasoning': sample['reasoning']
            }
            
            # Generate false version for each question
            for q_num in range(1, 8):
                q_key = f'Q{q_num}'
                original_text = sample.get(q_key, '')
                
                if original_text:
                    try:
                        messages = create_false_statement_prompt(
                            original_text, 
                            question_types[q_key]
                        )
                        
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=0.7,  # Slightly higher temperature for more variation
                            max_tokens=256,
                            top_p=0.9,
                        )
                        
                        false_text = completion.choices[0].message.content.strip()
                        false_sample[f'{q_key}_false'] = false_text
                        false_sample[f'{q_key}_true'] = original_text  # Keep original for reference
                        
                    except Exception as e:
                        logging.error(f"Error generating false statement for {q_key}: {str(e)}")
                        false_sample[f'{q_key}_false'] = ""
                        false_sample[f'{q_key}_true'] = original_text
                
            false_data.append(false_sample)
            
            # Save intermediate results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(false_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Processing complete. Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")

def main():
    input_file = "True_statements/*.json"
    output_file = "False_statements/*.json"
    
    start_time = time.time()
    generate_false_statements(input_file, output_file)
    end_time = time.time()
    logging.info(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()