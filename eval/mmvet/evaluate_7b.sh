

file_path="/fs-computility/mllm/xinglong/qwen2vl_scalecap_dong/eval/mm-vet/answers/Main_Table_VLM7B-val-Prismllm-QwQ32b-1-t0.jsonl"
file_name=$(basename "$file_path" .jsonl)

echo "filename: ${file_name}"

cd eval/mmvet
python convert_answers.py

cd ../..

python eval/mmvet/evaluator.py --result-file eval/mmvet/results/${file_name}.json