import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList
)
from utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from datasets import load_dataset
from collections import Counter
import json
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument("--use_cot_prompt", action="store_true")
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16) 
    
    print('\nLoading dataset...')
    dataset = load_dataset('gsm8k', "main", split='test')
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]

    results = []
    for i in tqdm(range(datasize), desc='Evaluating'):
        example = dataset[i]
        if args.use_cot_prompt:
            input_text = "Q: {question}\nA: Let's think step by step.".format(question=example['question'])
        else:
            input_text = 'Q: ' + example['question'] + '\nA:'
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])

        # Define a stopping condition for generation
        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                    outputs = model.generate(**inputs, temperature=args.temp, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract the final answer from the model's output
                output_text = output_text.split("A:")[-1].strip() 
                model_answer = extract_predicted_answer(output_text)
                model_answers.append({'text': output_text, 'numeric': model_answer})
        else:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split("A:")[-1].strip() 
            model_answer = extract_predicted_answer(output_text)
            model_answers.append({'text': output_text, 'numeric': model_answer})

        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        results.append({
            'question': example['question'],
            'gold_answer_text': example['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        })
    
    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")
    
    results.append({'accuracy': cnt / total})

    os.makedirs('eval_results/zero_shot', exist_ok=True)
    
    model_name = args.model.split('/')[-1]
    result_file = f"eval_results/zero_shot/{model_name}"
    if args.use_cot_prompt:
        result_file += "_cot"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")
                

if __name__ == '__main__':
    main()


