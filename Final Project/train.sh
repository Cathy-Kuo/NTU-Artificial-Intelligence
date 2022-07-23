python3 run_summarization.py \
	--model_name_or_path google/mt5-small \
	--do_train --do_eval --do_predict \
	--train_file data/train.json \
	--validation_file data/dev.json \
	--test_file data/dev.json \
	--source_prefix "summarize: " \
	--text_column maintext --summary_column title \
    --output_dir $1 \
	--save_total_limit 5 \
    --per_device_train_batch_size=$2 \
    --per_device_eval_batch_size=4 \
	--gradient_accumulation_steps=$3 \
    --overwrite_output_dir \
	--max_source_length 256 \
	--max_target_length 64 \
    --predict_with_generate \
	$4 $5 $6 $7 $8 $9
