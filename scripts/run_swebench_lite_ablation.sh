# 8k context, c3 7b
#python -m swebench.harness.run_evaluation --predictions_path patches/SftC37bSwebench-7B_20241217_105200_round_appearance-ckpt-578-32-10.jsonl --max_workers 8 --run_id 7b-baseline-ee --exclude_completed False
#
# 16k context, c3 7b
python -m swebench.harness.run_evaluation --predictions_path  patches/Blobheart:c3-sweep-tp0w9f2d-j7so-fp16-32-10.jsonl --max_workers 8 --run_id 7b-baseline-ee --exclude_completed False

