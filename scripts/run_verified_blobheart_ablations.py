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

swebench_half_scrape_runs = [
    "9o6gy25k",
    "ze026y1q",
    "s36u41pp",
    "6rfkf6qv",
    "kitjyddf",
]

swebench_half_datacurve_runs = [
    "juc1q36g",
    "wsle0532",
    "bw6kkgbv",
    "sent9ynb",
    "jfj1g27t",
]

swebench_scrape_runs = [
    "l0gis24e",
    "peqb32w5",
    "ngz4fqu9",
    "cgc7zx3e",
    "o7mo7rk1",
]

swebench_datacurve_runs = [
    "y55iib7y",
    "rtpp1ooe",
    "9qj9tmic",
    "65de9xas",
    "o67llyrr",
]

swebench_111b_runs = [
    "6mv5zues",
]

swebench_cot_runs = [
    "command3-111b-d7ajyi7h-h4pc-synth",
]

datamixes = dict(
    swebench=swebench_runs,
    swebench_half=swebench_half_runs,
    swebench_sixty=swebench_sixty_runs,
    swebench_half_datacurve=swebench_half_datacurve_runs,
    swebench_half_scrape=swebench_half_scrape_runs,
    swebench_scrape=swebench_scrape_runs,
    swebench_datacurve=swebench_datacurve_runs,
    swebench_111b=swebench_111b_runs,
)


command_template = "python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path  patches/{model}.jsonl --max_workers 128 --run_id swebench-verified-ablations --exclude_completed False"

# Dictionary to store results
results = {
    '4o': [],
    "claude": [],
    'mistral': [],
    "qwen": [],
    "llama": [],
    'swebench': [],
    "swebench_half": [],
    "swebench_sixty": [],
    "swebench_half_scrape": [],
    "swebench_half_datacurve": [],
    "swebench_scrape": [],
    "swebench_datacurve": [],
    "swebench_111b": [],
    "swebench_cot": [],
}

for path in Path("patches").glob("swebench-verified-*"):
    #continue
    model_name = path.stem  # Gets filename without extension
    if Path(f"{model_name}.swebench-verified-ablations.json").exists():
        print(model_name, "exists")
        continue
    #if "4o" not in model_name: continue

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
            elif "claude" in model_name:
                results["claude"].append(resolved_instances)
            elif "4o" in model_name:
                results["4o"].append(resolved_instances)
            elif "Qwen" in model_name:
                results["qwen"].append(resolved_instances)
            elif "llama" in model_name:
                results["llama"].append(resolved_instances)
            elif "command3-111b-d7ajyi7h-h4pc-synth" in model_name:
                results["swebench_cot"].append(resolved_instances)
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
