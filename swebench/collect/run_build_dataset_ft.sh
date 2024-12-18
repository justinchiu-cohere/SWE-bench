#!/bin/bash



python build_dataset_ft.py \
    --instances_path "saved_tasks" \
    --output_path "scrape-12-11" \
    --eval_path "saved_tasks"


#python build_dataset_ft.py \
#    --instances_path "<path to folder containing task instance (raw) files>" \
#    --output_path "<path to folder to save finetuning dataset to>" \
#    --eval_path "<path to folder containing all evaluation task instances>"
