import os
from pathlib import Path
import subprocess
import json



command_template = "python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path  patches/{model}.jsonl --max_workers 8 --run_id swebench-verified-ablations --exclude_completed False"

# Dictionary to store results
results = {
    '4o': [],
    'mistral': [],
    'swebench': [],
}

for path in Path("patches").glob("swebench-verified-*"):
    model_name = path.stem  # Gets filename without extension
    if Path(f"{model_name}.swebench-verified-ablations.json").exists():
        print(model_name, "exists")
        continue

    command = command_template.format(model=model_name)
    
    print(f"Processing {model_name}")
    print(f"Running command: {command}")
    
    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {model_name}: {e}")

for path in Path("patches").glob("swebench-verified-*"):
    model_name = path.stem  # Gets filename without extension
    # Read and parse the JSON file
    try:
        with open(f"{model_name}.7b-swebench-ablations.json", "r") as f:
            data = json.load(f)
            resolved_instances = data['resolved_instances']
            
            # Categorize based on wandb_id
            if "mistral" in model_name:
                results["mistral"].append(resolved_instances)
            elif "4o" in model_name:
                results["4o"].append(resolved_instances)
            else:
                results["swebench"].append(resolved_instances)
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

# Calculate and print statistics
for category, scores in results.items():
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n{category.upper()} Results:")
        print(f"Individual scores: {scores}")
        print(f"Average resolved instances: {avg_score:.2f}")
        print(f"Number of models: {len(scores)}")
