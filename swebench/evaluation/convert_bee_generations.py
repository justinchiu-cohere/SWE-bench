import json
import re
import os
import subprocess
from pathlib import Path
from typing import Dict, List
from git import Repo
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
import gcsfs
import pandas as pd

from swebench.evaluation.aider_edit import do_replace


TOKEN = os.getenv("GITHUB_TOKEN")

def remove_metatext(x):
    return (
        re.sub("\d+,\d+:", "", x)
        .replace("<patch>", "")
        .replace("</patch>", "")
        .replace("```patch", "")
        .replace("```", "")
        .strip()
    )


def process_patch(x):
    patches = remove_metatext(x)
    chunks = re.split(r"(\-\-\-.*?\n\+\+\+.*?\n)", patches, flags=re.MULTILINE | re.DOTALL)
    named_chunks = [
        (re.findall(r"\+\+\+ (.*)", filename)[0], patch)
        for filename, patch in zip(chunks[:-1], chunks[1:])
        if filename[:3] == "---"
    ]
    return named_chunks


def make_repo(repo, repo_path, verbose=True, token="git"):
    repo_url = (
        f"https://{token}@github.com/swe-bench/" + repo.replace("/", "__") + ".git"
    )
    repo_url = f"https://{token}@github.com/{repo}.git"
    if verbose:
        print(f"Cloning {repo} to {repo_path}")
    Repo.clone_from(repo_url, repo_path)


def init_repo_cache(dataset, root_dir, cloud_path, verbose):
    for instance_id, instance in dataset.items():
        repo_path = root_dir / instance["repo"].replace("/", "__")
        repo_path = repo_path.resolve()
        if not repo_path.exists():
            make_repo(instance, repo_path, verbose=verbose, token=TOKEN)


def apply_diff(repo_dir, diff):
    edits = re.findall(
        r"### (.*?)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        diff,
        flags=re.MULTILINE | re.DOTALL,
    )
    for filename, search, replace in edits:
        fname = repo_dir / filename
        if not fname.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
            fname.touch()
        content = ""
        with fname.open("r") as f:
            content += f.read()
        new_content = do_replace(fname, content, search, replace)
        if new_content is not None:
            # Successfully applied diff
            with fname.open("w") as f:
                f.write(new_content)
        else:
            print("DIFF NOT APPLIED")


def apply_diff_and_get_patch(repo, base_commit, instance_id, diff, repo_cache_path):
    """Apply a diff, cloning the repo if necessary, then git diff to get the patch"""

    base_repo_path = repo_cache_path / repo.replace("/", "__")
    if not base_repo_path.exists():
        print(f"Base repo {repo} was missing: Cloning")
        make_repo(repo, base_repo_path, verbose=True, token=TOKEN)

    repo_path = repo_cache_path / "commit_cache" / instance_id
    repo_path = repo_path.resolve()
    if not repo_path.exists():
        p = subprocess.run(
            f"git worktree add -f ../commit_cache/{instance_id} {base_commit}",
            cwd=base_repo_path,
            shell=True,
            capture_output=True,
        )
        print(p.stdout)
        print(p.stderr)

    apply_diff(repo_path, diff)
    output = subprocess.check_output(f"git diff".split(), cwd=repo_path).decode("utf-8")

    # cleanup commit_cache
    shutil.rmtree(repo_path)

    return output


def extract_model_name(filepath: str) -> str:
    """Extract model name from GCS filepath."""
    if 'SftC37b' in filepath:
        # Define the regex pattern to match everything after 'SftC3' but before '.parquet'
        pattern = r"(SftC3.*)\.parquet"

        # Search for the pattern in the original string
        match = re.search(pattern, filepath)

        # Extract the matched group if any
        result = match.group(1) if match else "unknown"
        return result
    elif 'Blobheart' in filepath:
        return 'Blobheart'
    elif 'OpenAIAPI' in filepath:
        return 'GPT-4'
    elif 'command-22.1.1' in filepath:
        return 'command-22.1.1'
    return 'unknown'

def process_row(row, model_name):
    """Process a single row and return the result."""
    try:
        unidiff_patch = apply_diff_and_get_patch(
            repo=row["repo"],
            base_commit=row["base_commit"],
            instance_id=row["instance_id"],
            diff=row["diff"],
            repo_cache_path=Path("/home/justinchiu_cohere_com/format-pr/repo_cache"),
        )
        
        if unidiff_patch:
            return {
                "model_name_or_path": model_name,
                "instance_id": row["instance_id"],
                "model_patch": unidiff_patch
            }
    except Exception as e:
        print(f"Error processing instance {row['instance_id']}: {e}")
    return None

def process_parquet_file(pq_path: str, output_dir: Path):
    """Process a single parquet file and save results as JSONL."""
    print(f"Processing {pq_path}")
    model_name = extract_model_name(pq_path)
    output_path = output_dir / f"{model_name}.jsonl"
    
    df = pd.read_parquet(pq_path)
    results = []
    
    # Process in batches of 16
    batch_size = 16
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Submit batch of rows and collect in order
            futures = [
                executor.submit(process_row, row, model_name)
                for _, row in batch.iterrows()
            ]
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
    
    # Write results to JSONL
    with output_path.open('w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    # List all parquet files in the GCS bucket
    fs = gcsfs.GCSFileSystem()
    bucket_path = "gs://cohere-dev/justinchiu/swebench_lite_generations/"
    parquet_files = [f"gs://{f}" for f in fs.ls(bucket_path) if f.endswith('.parquet')]
    
    # Create output directory
    output_dir = Path("patches")
    output_dir.mkdir(exist_ok=True)
    
    # Process each parquet file
    for pq_path in parquet_files:
        process_parquet_file(pq_path, output_dir)
