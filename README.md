# Reasoning-Based Fine-Tuning Pipeline

A pipeline to fine-tune the nanbeige4.1-3B model using electrical engineering data. The main purpose of the project is to significantly reduce thinking traces and produce more concise answers. It uses Unsloth for faster training.

---

## Configuration

The pipeline is controlled by one file: ```config/config.json```. You do not need to change the code; just edit the JSON file to change your paths, model settings, and hyperparameters.

---

## Data Acquisition

This project utilizes the Stack Exchange Data Dump. Because Stack Exchange requires authentication for bulk access, follow these steps:

1. Register: Create an account at [electronics.stackexchange.com](https://electronics.stackexchange.com).
2. Locate the Collection: Go to your account -> Settings -> Access -> Data dump access.
3. Download Data Dump: Download the `electronics.stackexchange.com.7z` archive.
4. Placement: Extract all contents of the compressed file into a local folder.
5. Configure: Update the `data_dir` in your `config.json` to point to this folder.

---

## Pipeline Overview

1. Data Curation: Parses raw Stack Exchange XML dumps into structured Q&A pairs.
2. Data Enrichment: Uses a teacher model to enrich the raw Stack Exchange answers.
3. Trace Generation: Uses a teacher model to generate thinking traces.
4. Data Splitting: Splits data into training and evaluation sets.
5. Fine-Tuning: LoRA training using Unsloth.
6. Merge & Convert: Merges the adapter to the base model and quantizes it.
7. Benchmarking: Compares base and fine-tuned models on GSM8K, MMLU Electrical Engineering, and held-out EE Q&A pairs using exact-match accuracy and LLM-as-judge evaluation.

---

## Installation and Setup

1. Clone the repository:
   ```powershell
   git clone https://github.com/amoughnieh/nanbeige-reasoning-finetune.git
   cd nanbeige-reasoning-finetune
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Download the GGUF Converter:
   ```powershell
   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py" -OutFile "convert_hf_to_gguf.py"
   ```
---

## Workflow Execution

The commands below use settings from config/config.json. Any argument you type in the command line will override the config file.

### Phase 1: Data Prep
```
python 1_data_curation.py
```
```
python 2_data_enrichment.py --max_items 50
```
```
python 3_trace_generation.py --max_items 50
```
```
python 4_create_splits.py
```

### Phase 2: Training
```
python 5_run_training.py
```

### Phase 3: Merge
```
python 6_merge_and_convert.py --adapter_path lora_adapter_r01
```
### Phase 4: Benchmarking
```
python 7_run_benchmarks.py --benchmark gsm8k --max_items 50
```
```
python 7_run_benchmarks.py --benchmark mmlu_ee --max_items 50
```
```
python 7_run_benchmarks.py --benchmark ee_qa --max_items 50 --eval_jsonl data/ee_qa_eval.jsonl --model gemini-2.5-pro
```
---

## References

* Base Model: [Nanbeige 4.1 3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)
* Research Paper: [Nanbeige 4.1: Paper](https://arxiv.org/abs/2602.13367)
* Model & GGUF: [Huggingface](https://huggingface.co/amoughnieh/Nanbeige4.1-3B-EE-Reasoning-SFT)

---

## Important Notes
* Overrides: To change a setting for just one run, type it in the command line. The command line always wins over the config file.
* API Safety: Scripts 2 and 3 do not have a default max items in the config. You must type --max_items in the command line to control your API spending.
* Adapter Path: In Step 6, you must specify the folder of your new adapter using --adapter_path.
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
