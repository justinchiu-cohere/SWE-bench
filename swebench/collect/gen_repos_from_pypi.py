import json

# Function to parse the jsonl file and filter repositories
def parse_and_filter_jsonl(file_path, star_threshold=100, pull_threshold=100, max_rank=1000):
    repos = []
    
    # Open and read the jsonl file
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            # Filter based on given criteria
            if (
                data.get('rank') is not None and 
                data['rank'] < max_rank and 
                data.get('stars') is not None and 
                data['stars'] > star_threshold and 
                data.get('pulls') is not None and 
                data['pulls'] > pull_threshold and 
                data.get('github') is not None
            ):
                github_url = data['github']
                # Extracting the repo name from the GitHub URL
                repo_name = github_url.split('https://github.com/')[-1]
                repos.append(repo_name)
    
    return repos

# Function to format the bash script
def format_bash_script(repos):
    # Start with the shebang and define a bash array
    lines = ["#!/bin/bash", "", "repos=("]
    for repo in repos:
        lines.append(f"    '{repo}'")  # Add each repo to the array
    lines.append(")")

    # Add a loop to iterate over each repository
    lines.extend([
        "", 
        "for repo in \"${repos[@]}\"; do",
        "    python get_tasks_pipeline.py \\",
        "        --repos \"$repo\" \\",
        "        --path_prs saved_prs \\",
        "        --path_tasks saved_tasks",
        "done"
    ])
    return "\n".join(lines)

# Main execution
def main():
    file_path = "pypi_rankings.jsonl"
    bash_path = "run_pypi_tasks_pipeline.sh"
    filtered_repos = parse_and_filter_jsonl(file_path)
    if filtered_repos:
        bash_script = format_bash_script(filtered_repos)
        with open(bash_path, "w") as f:
            f.write(bash_script)
        print("Script written to", bash_path)
    else:
        print("No repositories matched the given criteria.")
        
if __name__ == "__main__":
    main()
