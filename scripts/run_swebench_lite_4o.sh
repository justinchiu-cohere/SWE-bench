#python -m swebench.harness.run_evaluation --predictions_path gpt-4o-2024-08-06.jsonl --max_workers 8 --run_id 4o-oracle-diff-agentless --exclude_completed False
#python -m swebench.harness.run_evaluation --predictions_path OpenAIAPI:gpt-4o-2024-08-06.jsonl --max_workers 8 --run_id 4o-oracle-diff-agentless --exclude_completed False
#python -m swebench.harness.run_evaluation --predictions_path OpenAIAPI:gpt-4o-2024-08-06-32.jsonl --max_workers 8 --run_id 4o-oracle-diff-agentless --exclude_completed False
python -m swebench.harness.run_evaluation --predictions_path OpenAIAPI:gpt-4o-2024-08-06-16.jsonl --max_workers 8 --run_id 4o-oracle-diff-agentless --exclude_completed False
