
#!/bin/bash

config_path="configs/base_config.json"
log_prefix="log"

usage() {
    echo "Usage: $0 [-c config_path] [-l log_prefix]"
    exit 1
}

while getopts "c:l:h" option; do
    case "${option}" in
        c)
            config_path=${OPTARG}
            ;;
        l)
            log_prefix=${OPTARG}
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

tokenizer_types=("bpe" "wordpiece" "unigram")

script_path="train.py"

for tokenizer_type in "${tokenizer_types[@]}"; do
    echo "Starting training for tokenizer type: $tokenizer_type on GPU $gpu"
    nohup python "$script_path" --config "$config_path" --tokenizer_type "$tokenizer_type" > "${log_prefix}_${tokenizer_type}.out" 2>&1

done

echo "All training processes have completed"
