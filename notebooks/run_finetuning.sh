#!/bin/bash

datasets=("CB" "WiC" "COPA" "QQP" "BoolQA" "RTE" "IMDB" "yelp" "amazon" "SST-2" "dbpedia" "agnews" "MultiRC" "yahoo")
source_datasets=("MNLI" "CB" "WiC" "COPA" "QQP" "BoolQA" "RTE" "IMDB" "yelp" "amazon" "SST-2" "dbpedia" "agnews" "MultiRC")

scale_factor=0.07

for i in "${!datasets[@]}"; do
    source_dataset="${source_datasets[$i]}"
    fine_tune_dataset="${datasets[$i]}"
    if [ "$fine_tune_dataset" == "CB" ]; then
        starting_checkpoint="t5_finetuned_mnli.pt"
    else
        starting_checkpoint="t5_svd_${source_dataset,,}.pt"
    fi
    output_model_name="t5_svd_${fine_tune_dataset,,}.pt"

    printf "Running fine-tuning on %s (starting from %s) with scale factor %.2f\n" "$fine_tune_dataset" "$source_dataset" "$scale_factor"

    python finetune_svd.py \
        --source_svd_dataset "$source_dataset" \
        --fine_tune_dataset "$fine_tune_dataset" \
        --starting_checkpoint "$starting_checkpoint" \
        --output_model_name "$output_model_name" \
        --scale_factor "$scale_factor"

    sleep 5  # Short delay to prevent excessive GPU resource allocation
    scale_factor=$(python -c "print(${scale_factor} + 0.07)")   # Increment scale factor
done