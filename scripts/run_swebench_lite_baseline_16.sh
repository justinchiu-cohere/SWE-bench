#python -m swebench.harness.run_evaluation --predictions_path baseline.jsonl --max_workers 8 --run_id refresh-oracle-diff-agentless-baseline --exclude_completed False
python -m swebench.harness.run_evaluation --predictions_path gs:--co-train-oktodelete-90d-justinchiu_cohere_com-command-20.1.0-dev-35B_20241022_044459_most_audience-ckpt-0-16.jsonl --max_workers 8 --run_id refresh-oracle-diff-agentless-baseline --exclude_completed False