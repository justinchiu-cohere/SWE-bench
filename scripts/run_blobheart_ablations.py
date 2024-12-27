import os
from pathlib import Path
import subprocess
import json

baseline_wandb_runs = [
    "27omokby",
    "ampni7sp",
    "ab69328y",
    "xyh2fp1v",
    "5pqaxf1r",
]
swebench_wandb_runs = [
    "tp0w9f2d",
    "ofrxv6ke",
    "4yku4e2z",
    "4jtt1k9l",
    "4lbw861n",
    "zkey7k3u",
    "1tb0s94i",
    "gfh0o04h",
    "zf7cubyt",
    "8bk66u7l",
]
swebench_scrape_wandb_runs = [
    "1meyqtnm",
    "mwbbqv5c",
    "wsigy7al",
    "qlgqof20",
    "bowyk271",
]
swebench_half_wandb_runs = [
    "dkg32sen",
    "p4qou408",
    "g7m1aajx",
    "h5d8l7kw",
    "8o5f5ls9",
]
swebench_datacurve_wandb_runs = [
    "auyqf4ak",
    "18u29baj",
    "8k0o3hw2",
    "jezb8c0r",
    "d3109o6t",
]

command_template = "python -m swebench.harness.run_evaluation --predictions_path  patches/{model}.jsonl --max_workers 8 --run_id 7b-swebench-ablations --exclude_completed False"

# Dictionary to store results
results = {
    'baseline': [],
    'swebench': [],
    'swebench_scrape': [],
    'swebench_half': [],
    'swebench_datacurve': []
}

for path in Path("patches").glob("Blobheart*"):
    # dont run on non-sweep runs
    if "sweep" not in str(path):
        continue

    model_name = path.stem  # Gets filename without extension
    if Path(f"{model_name}.7b-swebench-ablations.json").exists():
        continue

    command = command_template.format(model=model_name)
    
    print(f"Processing {model_name}")
    print(f"Running command: {command}")
    
    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {model_name}: {e}")

for path in Path("patches").glob("Blobheart*"):
    # dont run on non-sweep runs
    if "sweep" not in str(path):
        continue

    model_name = path.stem  # Gets filename without extension
    # Read and parse the JSON file
    wandb_id = model_name.split('sweep-')[1].split('-')[0]
    try:
        with open(f"{model_name}.7b-swebench-ablations.json", "r") as f:
            data = json.load(f)
            resolved_instances = data['resolved_instances']
            
            # Categorize based on wandb_id
            if wandb_id in baseline_wandb_runs:
                results['baseline'].append(resolved_instances)
            elif wandb_id in swebench_wandb_runs:
                results['swebench'].append(resolved_instances)
            elif wandb_id in swebench_scrape_wandb_runs:
                results['swebench_scrape'].append(resolved_instances)
            elif wandb_id in swebench_half_wandb_runs:
                results['swebench_half'].append(resolved_instances)
            elif wandb_id in swebench_datacurve_wandb_runs:
                results['swebench_datacurve'].append(resolved_instances)
            else:
                results['unknown'].append(resolved_instances)
                print(f"Unknown wandb_id: {wandb_id}")
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
