import json
import os
import openai
from openai import OpenAI
from time import sleep
from tqdm import tqdm

# Set your OpenAI API key
# Initialize OpenAI client
client = OpenAI(api_key='')

def rephrase_prompt(text):
    """
    Use GPT to generate 4 rephrased versions of the text
    """
    system_prompt = """Yaert. Your task is to rephrase the given text 4 different times. 
    Important requirements:
    1. Maintain ALL medical details, numbers, and specific information
    2. Keep the same meaning and medical accuracy
    3. Use different sentence structures, tone, and wording to simulate diverse user prompts
    4. Each version should vary in:
       - Formality level (formal to conversational)
       - Sentence structure (simple to complex)
       - Vocabulary choice (technical to lay terms, while maintaining accuracy)
       - Writing style (direct, descriptive, narrative, etc.)
    5. Return ONLY the 4 rephrased versions, numbered 1-4
    
    Format your response exactly like this:
    1. [First rephrased version]
    2. [Second rephrased version]
    3. [Third rephrased version]
    4. [Fourth rephrased version]"""

    user_prompt = f"Please rephrase this medical text 4 times:\n\n{text}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",  # or your preferred model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Split into individual versions and clean up
        versions = []
        for line in response_text.split('\n'):
            if line.strip() and line[0].isdigit():
                # Remove the number and dot at the start
                clean_text = line.split('. ', 1)[1].strip()
                versions.append(clean_text)
        
        return versions
    
    except Exception as e:
        print(f"Error in GPT API call: {e}")
        return None

def process_nct_files(input_dir, output_dir):
    """
    Process only NCT files that don't exist in the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all input NCT files
    input_files = [f for f in os.listdir(input_dir) if f.startswith('NCT') and f.endswith('.json')]
    
    # Get list of already processed files
    existing_files = set(os.listdir(output_dir))
    
    # Filter files that need processing
    files_to_process = [f for f in input_files if f not in existing_files]
    
    print(f"Total input files: {len(input_files)}")
    print(f"Already processed: {len(input_files) - len(files_to_process)}")
    print(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print("No new files to process.")
        return
    
    for nct_file in tqdm(sorted(files_to_process), desc="Processing NCT files"):
        input_path = os.path.join(input_dir, nct_file)
        output_path = os.path.join(output_dir, nct_file)
        
        try:
            # Read the original file
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            original_prompt = data['prompt']
            
            # Get rephrased versions
            rephrased_versions = rephrase_prompt(original_prompt)
            
            if rephrased_versions:
                # Create new data structure with all versions
                data['prompts'] = [original_prompt] + rephrased_versions
                del data['prompt']  # Remove the original single prompt
                
                # Save the updated file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"Successfully processed: {nct_file}")
                sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error processing {nct_file}: {e}")
            continue

if __name__ == "__main__":
    input_directory = "Database_free_evaluation/HealthBench/Data/data"
    output_directory = "Database_free_evaluation/HealthBench/Data/data"
    
    process_nct_files(input_directory, output_directory)
