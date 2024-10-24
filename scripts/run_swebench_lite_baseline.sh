#python -m swebench.harness.run_evaluation --predictions_path baseline.jsonl --max_workers 8 --run_id refresh-oracle-diff-agentless-baseline --exclude_completed False
python -m swebench.harness.run_evaluation --predictions_path Blobheart:command-r-plus-v2-synth-a100-gg-32.jsonl --max_workers 8 --run_id refresh-oracle-diff-agentless-baseline --exclude_completed False
