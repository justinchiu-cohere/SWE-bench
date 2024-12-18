import json
import re
import os
import subprocess
from pathlib import Path
from typing import Dict, List
from git import Repo

import datasets
import gcsfs
import pandas as pd

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
    import pdb; pdb.set_trace()
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
        return new_content is not None


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
    import pdb; pdb.set_trace()
    return output


if __name__ == "__main__":
    pq_suffix = "-data-15d-post-training-justinchiu_cohere_com-code-SftC37bCode-7B_20241217_055657_clean_expression-ckpt-552-32-10.parquet"
    pq_path = f"gs://cohere-dev/justinchiu/swebench_lite_generations/{pq_suffix}"

    df = pd.read_parquet(pq_path)
    for idx, row in df.iterrows():
        unidiff_patch = apply_diff_and_get_patch(
            repo=row["repo"],
            base_commit=row["base_commit"],
            instance_id=row["instance_id"],
            diff=row["diff"],
            repo_cache_path=Path(f"/home/justinchiu_cohere_com/format-pr/repo_cache"),
        )
        import pdb; pdb.set_trace()
