import os
import json
import transformers
import torch
from vllm import LLM, SamplingParams
import time
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Constants
MODEL_PATH = "Llama-3.3-70B-Instruct/"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.85

# Pattern definitions
YES_PATTERNS = [
    "yes",
    "**yes**",
    "the statement is supported by the context",
    "is indeed supported",
]

NO_PATTERNS = [
    "no.",
    "**no**",
    "the statement is not supported by the context",
    "statement is not fully supported",
]

DIRECT_PATTERNS = {
    "answer: yes": 0,
    "answer: no": 1
}

def load_model() -> LLM:
    """Initialize and return the LLM model."""
    print("Loading model...")
    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.92,
        max_num_batched_tokens=2048,
    )

def map_evaluation_to_score(evaluation: str) -> Dict[str, Any]:
    """
    Map the evaluation text to scores based on pattern matches.
    
    Args:
        evaluation: The evaluation text to analyze
        
    Returns:
        Dictionary containing scoring details and metrics
    """
    eval_lower = evaluation.lower()
    
    # Count matches for each category
    yes_matches = sum(1 for pattern in YES_PATTERNS if pattern in eval_lower)
    no_matches = sum(1 for pattern in NO_PATTERNS if pattern in eval_lower)

    # Check for direct answer matches
    direct_score = 0.5
    for pattern, score in DIRECT_PATTERNS.items():
        if pattern in eval_lower:
            direct_score = score
            break
    
    # Create result dictionary
    result = {
        'yes_count': yes_matches,
        'no_count': no_matches,
        'total_matches': yes_matches + no_matches,
        'confidence': 'low',
        'direct_score': direct_score,
    }
    
    # Determine majority score
    if yes_matches == 0 and no_matches == 0:
        result.update({'score': 0.5, 'decision_type': 'no_matches'})
    elif yes_matches > no_matches:
        result.update({
            'score': 0,
            'decision_type': 'yes_majority',
            'confidence': 'high' if yes_matches > 1 else 'medium'
        })
    elif no_matches > yes_matches:
        result.update({
            'score': 1,
            'decision_type': 'no_majority',
            'confidence': 'high' if no_matches > 1 else 'medium'
        })
    else:
        result.update({
            'score': 0.5,
            'decision_type': 'tie',
            'confidence': 'medium' if yes_matches > 0 else 'low'
        })
    
    return result

def create_prompt_factuality_evaluation(context: str, sentence: str) -> str:
    """
    Create a prompt for factuality evaluation.
    
    Args:
        context: The context text
        sentence: The statement to evaluate
        
    Returns:
        Formatted prompt string
    """
    return (
        "You are an advanced clinical language model. Your task is to answer whether "
        "a statement is supported by a given context.\n\n"
        "The **statement**.\n\n"
        f"STATEMENT:{sentence}\n\n"
        "The **context**.\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Is the statement supported by the context above?. Answer Yes or No.\n\n"
        "ANSWER:"
        "Explain your decision:"
    )

def process_files(
    llm: LLM,
    statement_folder: str,
    context_folder: str,
    output_folder: str
) -> None:
    """Process all files, with batched questions within each file."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        statement_files = [f for f in os.listdir(statement_folder) if f.endswith('.json')]
        total_files = len(statement_files)
        print(f"Found {total_files} files to process")
        
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_NEW_TOKENS
        )
        
        # Process files with progress bar
        for filename in tqdm(statement_files, desc="Processing files", unit="file"):
            process_single_file_batch(
                llm=llm,
                filename=filename,
                statement_folder=statement_folder,
                context_folder=context_folder,
                output_folder=output_folder,
                sampling_params=sampling_params
            )
            
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"Error in process_files: {str(e)}")

def process_single_file_batch(
    llm: LLM,
    filename: str,
    statement_folder: str,
    context_folder: str,
    output_folder: str,
    sampling_params: SamplingParams
) -> None:
    """Process all questions in a single file as one batch."""
    try:
        # Load files
        with open(os.path.join(statement_folder, filename), "r", encoding='utf-8') as f:
            statement_data = json.load(f)
        
        with open(os.path.join(context_folder, filename), "r", encoding='utf-8') as f:
            context_data = json.load(f)
        
        context = context_data["Final_text"]
        
        # Collect all prompts for this file
        prompts = []
        question_keys = []
        
        # Create prompts for all questions
        for question_number in range(1, 16):
            key = f"Q{question_number}"
            if key in statement_data:
                prompt = create_prompt_factuality_evaluation(
                    context, statement_data[key]
                )
                prompts.append(prompt)
                question_keys.append(key)
        
        # Process all prompts in one batch
        file_results = {}
        if prompts:
            outputs = llm.generate(prompts, sampling_params)
            
            # Process results
            for i, output in enumerate(outputs):
                key = question_keys[i]
                evaluation_text = output.outputs[0].text
                
                # Get scoring results
                score_result = map_evaluation_to_score(evaluation_text)
                
                # Store results
                file_results[key] = {
                    'statement': statement_data[key],
                    'evaluation': evaluation_text,
                    'score': score_result['score'],
                    'direct_score': score_result['direct_score'],
                    'yes_matches': score_result['yes_count'],
                    'no_matches': score_result['no_count'],
                    'total_matches': score_result['total_matches'],
                    'decision_type': score_result['decision_type'],
                    'confidence': score_result['confidence']
                }
        
        # Save results for this file
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_results, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")

def main():
    """Main execution function."""
    try:
        statement_folder = "Inference_json"
        context_folder = "Files_with_Summary"
        output_folder = "Evaluation_json"
        
        llm = load_model()
        
        # Track total execution time
        start_time = time.time()
        process_files(llm, statement_folder, context_folder, output_folder)
        total_time = time.time() - start_time
        
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()