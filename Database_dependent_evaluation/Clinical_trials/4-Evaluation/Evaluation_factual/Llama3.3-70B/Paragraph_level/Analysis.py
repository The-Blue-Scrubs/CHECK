def analyze_scores(folder_path: str):
    """
    Read and analyze scores from all JSON files in the folder.
    
    Args:
        folder_path: Path to folder containing JSON files
    
    Returns:
        dict: Sum of scores for each question
        int: Total number of files processed
    """
    # Initialize score sums for Q1-Q15
    direct_score_sums = {f"Q{i}": 0 for i in range(1, 16)}
    score_sums = {f"Q{i}": 0 for i in range(1, 16)}
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    total_files = len(json_files)
    
    print(f"Found {total_files} files to process")
    
    # Process each file
    for filename in tqdm(json_files, desc="Processing files"):
        try:
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Sum scores for each question
                for q_num in range(1, 16):
                    key = f"Q{q_num}"
                    if key in data and 'aggregate_direct_score' in data[key]:
                        direct_score_sums[key] += data[key]['aggregate_direct_score']
                        score_sums[key] += data[key]['aggregate_score']    
                        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return direct_score_sums, score_sums, total_files

def plot_scores(score_sums: dict, total_files: int, name: str):
    """
    Create a bar plot of the scores.
    
    Args:
        score_sums: Dictionary containing sum of scores for each question
        total_files: Total number of files processed
        name: Name for saving the plot
    """

    x_axes_names = {
        1: "Title and purpose.",
        2: "Condition studied.",
        3: "Design Details.",
        4: "Interventions.",
        5: "Study Arms.",
        6: "Eligibility Criteria.",
        7: "Primary Outcome measured.",
        8: "Primary Outcome Statistical Analysis.",
        9: "Primary Outcome Statistical Results.",
        10: "Secondary Outcomes Overview.",
        11: "Secondary Outcomes Statistical Approach",
        12: "Secondary Outcomes Key Results.",
        13: "Serious Adverse Events.",
        14: "Non-Serious Adverse Events. ",
        15: "Key Observations and Clinical Relevance."
    }
    
    # Prepare data for plotting
    questions = list(range(1, 16))
    scores = [score_sums[f"Q{i}"] for i in questions]
    
    # Create figure and axis with larger size
    plt.figure(figsize=(20, 10))
    
    # Create bar plot
    bars = plt.bar(questions, scores)
    
    # Customize plot
    plt.title('Sum of Scores by Question Category', fontsize=16, pad=20)
    plt.xlabel('Question Categories', fontsize=14)
    plt.ylabel('Sum of Scores', fontsize=14)
    
    # Set x-axis ticks with rotated labels
    plt.xticks(questions, 
               [x_axes_names[i] for i in questions],
               rotation=45,
               ha='right',
               fontsize=12)
    
    # Set y-axis limit from 0 to total number of files
    plt.ylim(0, total_files)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', 
                va='bottom',
                fontsize=10)
    
    # Add some padding at the bottom for the rotated labels
    plt.subplots_adjust(bottom=0.2)
    
    # Save plot
    plt.savefig(name, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{name}'")
    
    # Show plot
    plt.show()
    
    # Close the figure to free memory
    plt.close()