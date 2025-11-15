import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel
import argparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dataset_utils import load_dataset, canonicalize_dataset_name

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = canonicalize_dataset_name(args.dataset_name)
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, api_base=args.api_base_url)
        self.prompt_creator = self.prompt_LSAT
        self.label_phrase = 'The correct option is:'
    
    def prompt_LSAT(self, in_context_example, test_example):
        full_prompt = in_context_example
        context = test_example['context'].strip()
        question = test_example['question'].strip()
        options = '\n'.join([opt.strip() for opt in test_example['options']])
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[OPTIONS]]', options)
        return full_prompt

    def load_in_context_examples(self):
        with open(os.path.join(self.demonstration_path, f'{self.dataset_name}_{self.mode}.txt')) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_raw_dataset(self, split):
        return load_dataset(self.dataset_name, self.data_path, split)

    def reasoning_graph_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()
        
        outputs = []
        for example in tqdm(raw_dataset):
            question = example['question']

            # create prompt
            full_prompt = self.prompt_creator(in_context_examples, example)
            output = self.openai_api.generate(full_prompt)
            if output is None:
                print(f"[Baseline] Skipping example {example['id']} due to API error.")
                outputs.append(self.failed_output(example))
                continue

            # get the answer
            label_phrase = self.label_phrase
            generated_answer = output.split(label_phrase)[-1].strip()
            generated_reasoning = output.split(label_phrase)[0].strip()

            # create output
            output = {'id': example['id'], 
                      'question': question, 
                      'answer': example['answer'], 
                      'predicted_reasoning': generated_reasoning,
                      'predicted_answer': generated_answer}
            outputs.append(output)

        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_creator(in_context_examples, example) for example in chunk]
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    if output is None:
                        print(f"[Baseline] Skipping example {sample['id']} due to API error.")
                        outputs.append(self.failed_output(sample))
                        continue
                    # get the answer
                    dict_output = self.update_answer(sample, output)
                    outputs.append(dict_output)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.openai_api.generate(full_prompt)
                        if output is None:
                            print(f"[Baseline] Skipping example {sample['id']} due to API error.")
                            outputs.append(self.failed_output(sample))
                            continue
                        # get the answer
                        dict_output = self.update_answer(sample, output)
                        outputs.append(dict_output)
                    except:
                        print('Error in generating example: ', sample['id'])

        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
    def update_answer(self, sample, output):
        text = output.strip() if output is not None else ''
        reasoning = text
        answer_segment = text

        # try to split by known markers for the final answer
        markers = [
            self.label_phrase,                    # "The correct option is:"
            'The correct answer is:',
            '选项：',
            '选项:',
        ]
        for marker in markers:
            if marker in text:
                before, after = text.split(marker, 1)
                reasoning = before.strip()
                answer_segment = after.strip()
                break

        # extract a short choice string (e.g., "A", "B", ...)
        generated_answer = self._extract_choice(answer_segment)
        if generated_answer is None:
            # fallback: keep the raw tail as answer
            generated_answer = answer_segment

        dict_output = {
            'id': sample['id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'predicted_reasoning': reasoning,
            'predicted_answer': generated_answer,
        }
        return dict_output

    def _extract_choice(self, answer_str: str):
        s = (answer_str or '').strip()
        if not s:
            return None
        # normalize some simple wrappers, e.g. "(D)", "D)", "D."
        first = s[0].upper()
        if first in 'ABCDEFGH':
            return first
        # also look at the last non-space character
        last = s[-1].upper()
        if last in 'ABCDEFGH':
            return last
        return None

    def failed_output(self, sample):
        return {'id': sample['id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'predicted_reasoning': 'API_ERROR',
                'predicted_answer': ''}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--api_base_url', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.batch_reasoning_graph_generation(batch_size=10)
