import os
import json
import random 
import time
from tqdm import tqdm
import logging
from openai import OpenAI
from typing import List

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

def create_prompt(data, question_number):
    """Creates a prompt for a specific question number"""
    questions = {
        1: "Definition and Pathophysiology: Define the disorder and outline its underlying causes and mechanisms.",
        2: "Risk Factors: List key factors that increase the likelihood of developing this disorder.",
        3: "Symptoms: Describe the disorder primary symptoms and explain why these symptoms occur based on the disorder's pathophysiology.",
        4: "Tests: Identify essential diagnostic tests or imaging methods that confirm this disorder.",
        5: "Differentiation: Explain how to distinguish this disorder from other disorders with similar presentations.",
        6: "First-Line Treatment: Describe the recommended initial treatment and explain how the treatment addresses the disorder.",
        7: "Complications: List potential complications related to the disorder.",
    }
    
    system_message = (
        "You are an advanced clinical language model. Use medically precise terminology. "
        "Be specific and focused ONLY on answering the requested question. "
        "Format your response as a cohesive paragraph."
        "Avoid any introductory or concluding sentences."
    )
    
    user_message = (
        f"Use the provided information to answer the following question about the disorder: {data}\n\n"
        f"QUESTION:\n{questions[question_number]}\n\n"
        "Remember to format your response as a single, well-structured paragraph."
    )
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def process_samples(input_file: str, output_file: str, num_samples: int = 2):
    """Process one random sample at a time, sending all questions as a batch"""
    
    try:
        # Read input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Convert json_data to list if it's a dictionary and keep track of indices
        if isinstance(json_data, dict):
            indices = list(json_data.keys())
            json_data = list(json_data.values())
        else:
            indices = list(range(len(json_data)))
        
        # Get random samples and their indices
        sample_indices = random.sample(range(len(json_data)), num_samples)
        results = []
        
        # Process one sample at a time
        for idx, sample_index in enumerate(tqdm(sample_indices, desc="Processing samples")):
            logging.info(f"\nProcessing sample {idx + 1}/{num_samples}")
            
            sample_data = json_data[sample_index]
            original_index = indices[sample_index]
            
            # Create result entry
            result = {
                'index': original_index,
                'AUI': sample_data.get('AUI', ''),
                'CUI': sample_data.get('CUI', ''),
                'term': sample_data.get('term', ''),
                'Definition': sample_data.get('Definition', ''),
                'reasoning': sample_data.get('reasoning', '')
            }
            
            # Process all questions for this sample
            for q_num in range(1, 8):
                try:
                    prompt_text = sample_data.get('term', '') + sample_data.get('reasoning', '')
                    messages = create_prompt(prompt_text, q_num)
                    
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        top_p=0.85,
                    )
                    
                    result[f'Q{q_num}'] = completion.choices[0].message.content.strip()
                    
                except Exception as e:
                    logging.error(f"Error processing Q{q_num} for sample {idx + 1}: {str(e)}")
                    result[f'Q{q_num}'] = ""
            
            results.append(result)
            
            # Save intermediate results after each sample
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
        logging.info(f"\nProcessing complete. Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")

def main():
    input_file = "True_statements/*.json"
    output_file = "True_statements/*.json"
    
    start_time = time.time()
    process_samples(input_file, output_file)
    end_time = time.time()
    logging.info(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()