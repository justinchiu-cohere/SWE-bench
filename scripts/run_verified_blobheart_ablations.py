import os
from pathlib import Path
import subprocess
import json
import re


swebench_runs = [
    "j1y3d6j2",
    "eerhmki0",
    "fghlpdtv",
    "4bvp3gb9",
    "ldz3g9wd",
]

swebench_half_runs = [
    "xwe9zchs",
    "jqzlqtiq",
    "4sf3qwe9",
    "pp6vtv1r",
    "pj4n7clf",
]

swebench_sixty_runs = [
    "11wsdi8b",
    "gwr7pdh4",
    "3csh0vmd",
    "jhd0snee",
    "b1qzt0w7",
]

swebench_half_datacurve_runs = [
]

datamixes = dict(
    swebench=swebench_runs,
    swebench_half=swebench_half_runs,
    swebench_sixty=swebench_sixty_runs,
    swebench_half_datacurve=swebench_half_datacurve_runs,
)


command_template = "python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path  patches/{model}.jsonl --max_workers 8 --run_id swebench-verified-ablations --exclude_completed False"

# Dictionary to store results
results = {
    '4o': [],
    'mistral': [],
    'swebench': [],
    "swebench_half": [],
    "swebench_sixty": [],
    "swebench_half_datacurve": [],
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
        with open(f"{model_name}.swebench-verified-ablations.json", "r") as f:
            data = json.load(f)
            resolved_instances = data['resolved_instances']
            
            # Categorize based on wandb_id
            if "mistral" in model_name:
                results["mistral"].append(resolved_instances)
            elif "4o" in model_name:
                results["4o"].append(resolved_instances)
            elif "c3-sweep" in model_name:
                # names look like: "swebench-verified-c3-sweep-jqzlqtiq-xuse-fp16-32-10.parquet"
                # 1. get the wandb
                wandb_id = re.search(r"c3-sweep-(.*?)-", model_name).group(1)
                # 2. find the datamix corresponding to the wandb id
                for datamix, ids in datamixes.items():
                    if wandb_id in ids:
                        results[datamix].append(resolved_instances)
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
