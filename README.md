# eval_gsm8k

This repository offers a lightweight and flexible solution for evaluating models on the GSM8K benchmark. The results are generally consistent with those obtained using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k).

## few-shot
### 8-shot
The 8-shot prompt is from the [lm-evaluation-harness gsm8k-cot](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml)

`python eval_gsm8k_few_shot.py --model <model_name>`

| Model           | Accuracy | Harness Accuracy |
|-----------------|----------|------------------|
| Mistral-7B-v0.1 | 41.02    | 42.99 (1.36)     |
| Llama-2-7b-hf   | 13.72    | 14.33 (0.97)     |

### 8-shot maj1@8

`python eval_gsm8k_zero_shot.py --model <model_name> --use_majority_vote --temp 0.2 --n_votes 8`
| Model           | Accuracy | Harness Accuracy |
|-----------------|----------|------------------|
| Mistral-7B-v0.1 | 47.84    | 44.96 (1.37)     |

`python eval_gsm8k_zero_shot.py --model <model_name> --use_majority_vote --temp 0.4 --n_votes 8`
| Model           | Accuracy |
|-----------------|----------|
| Mistral-7B-v0.1 | 50.57    |

# zero-shot
## cot zero-shot
use the Chain of Thought prompt "Let's think step by step." before answering the question.

`python eval_gsmk_zero_shot.py --model <model_name> --use_cot_prompt`

| Model           | Accuracy | Harness Accuracy |
|-----------------|----------|------------------|
| Mistral-7B-v0.1 | 22.06    | 15.85 (1.01)     |

## zero-shot

`python eval_gsmk_zero_shot.py --model <model_name>`

| Model           | Accuracy |
|-----------------|----------|
| Mistral-7B-v0.1 | 10.31    |







