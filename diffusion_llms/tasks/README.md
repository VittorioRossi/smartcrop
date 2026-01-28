# Evaluation Tasks for LLaDA

This directory contains custom evaluation task configurations for LLaDA.

## Available Tasks

### SimpleQA (`simpleqa_local`)
- **Source**: Local CSV file at `local_data/simple_qa_test_set.csv`
- **Task**: Question answering with exact match evaluation
- **Metric**: Exact match (case and punctuation insensitive)
- **Usage**: Tests factual knowledge and short-form QA capabilities

### LongFormQA (`longformqa_eli5`)
- **Source**: ELI5 dataset from HuggingFace
- **Task**: Long-form question answering
- **Metric**: ROUGE-L
- **Usage**: Tests ability to generate detailed, informative explanations

### WMT14 Translation
- **Source**: Built-in lm-evaluation-harness tasks
- **Tasks**: `wmt14_en-fr`, `wmt14_fr-en`
- **Metric**: BLEU, ChrF, TER
- **Usage**: Tests translation capabilities between English and French

## Usage

### SLURM Jobs
Run the evaluation jobs on HPC:

```bash
# SimpleQA evaluation
sbatch jobs/slurm/simpleqa-llada.sh

# LongFormQA evaluation  
sbatch jobs/slurm/longformqa-llada.sh

# WMT14 translation evaluation
sbatch jobs/slurm/wmt14-llada.sh
```

### Local Testing
For local testing, you can run the evaluation directly:

```bash
# Include path for custom tasks and run SimpleQA
accelerate launch diffusion_llms/eval_llada.py \
  --include_path eval_tasks \
  --tasks simpleqa_local \
  --model llada_dist \
  --batch_size 8 \
  --model_args "model_path='GSAI-ML/LLaDA-8B-Base',gen_length=64,steps=64,block_length=64,remasking='low_confidence',is_check_greedy=False" \
  --output_path results/simpleqa_test.json
```

## File Structure

```
eval_tasks/
├── __init__.py
├── simpleqa/
│   ├── __init__.py
│   └── simpleqa.yaml
└── longformqa/
    ├── __init__.py
    └── longformqa.yaml
```

## Requirements

These evaluations require the following Python packages:
- `evaluate`
- `rouge-score` 
- `sacrebleu`
- `datasets`

The SLURM scripts automatically install these if missing.
