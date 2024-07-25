import csv
import json
import argparse
import os
import torch
import numpy as np
import random

import time
import re
from tqdm import tqdm
from distutils.util import strtobool
import logging
import sys
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
# Use docker hosted VLLM for inference
import jsonlines
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY" # Can be anything
openai_api_base = "http://10.204.100.70:11700/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def create_json_file(destination):

    if not os.path.exists(destination):
        os.makedirs(destination)

    # Construct the full path to the JSON file
    file_name = destination.split("/")[-1]
    json_file_path = os.path.join(destination, file_name)+".jsonl"

    # Check if the JSON file already exists
    # if not os.path.exists(json_file_path):
    #     # Create the JSON file with an empty dictionary
    #     with open(json_file_path, 'w') as f:
    #         json.dump({}, f)
    # If the file doesn't exist, create it with mode 'w'
    if not os.path.exists(json_file_path):
        with jsonlines.open(json_file_path, mode='w') as writer:
            pass  # Create an empty file

    print(f"JSON file created at: {json_file_path}")

MNLI_label = ['entailment', 'neutral', 'contradiction']

model_name = "google/gemma-2-9b-it"

def load_evalset(args):
    dataset = load_dataset("json", data_files=args.dataset)
    test_df = dataset["train"]
    return test_df

# Function to clean the output
def clean_output(text):
    cleaned = re.sub(r'<\|im_(start|end)\|>', '', text)
    cleaned = re.sub(r'\n+', '\n', cleaned).strip()
    first_response = cleaned.split('assistant')[0].strip()
    return first_response if first_response else cleaned

# Function to process a single prompt
def generate(prompt):
    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": prompt},
    ],
    max_tokens=200,
    temperature=0.0,
    stop=["<|im_end|>"]
    )
    return completion.choices[0].message.content

def format_generation(example):

    prompt_template = """Context:{}
    Question:{}
    Answer the question as either {}, {}, or {}. Answer in the JSON format with the key 'answer'.""".format(
        example['context'],
        example['question'],
        example['ans0'],
        example['ans1'],
        example['ans2']
    )

    answer_list = [example['ans0'].lower(),example['ans1'].lower(),example['ans2'].lower()]
    return prompt_template, answer_list

def parse_json_answer(response):
    try:
        # Extract JSON from the response
        # print("response: ",response)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed.get('answer', '').lower()
    except:
        pass
    return None

def score_the_response(prediction, label, answer_list):
    return prediction == answer_list[label].lower()

def append_to_json(file_path, new_data):
    with jsonlines.open(file_path, mode='a') as writer:
        writer.write(new_data)

def evaluate(output_path, dataset):
    correct = 0
    total = 0

    prediction_list = []
    label_list = []

    create_json_file(output_path)

    dataset = dataset.shuffle(seed = 42)
    for example in tqdm(dataset):
        # Generate responses for the batch
        prompt, answer_list = format_generation(example)
        response = generate(prompt)
        results = []
        # Process each response
        prediction = parse_json_answer(response)
        # print("prediction: ", prediction)
        
        if prediction is not None:
            is_correct = score_the_response(prediction.lower(), example['label'], answer_list)
            correct += int(is_correct)
        total += 1

        # results.append({
        #     'context': example['context'],
        #     'question': example['question'],
        #     'label': example['label'],
        #     'prediction': prediction,
        #     'response': response,
        #     'acc': int(is_correct if prediction is not None else False),
        #     'answer 0': example['ans0'],
        #     'answer 1': example['ans1'],
        #     'answer 2': example['ans2'],
        # })
        results = {
            'context': example['context'],
            'question': example['question'],
            'label': example['label'],
            'prediction': prediction,
            'response': response,
            'acc': int(is_correct if prediction is not None else False),
            'answer 0': example['ans0'],
            'answer 1': example['ans1'],
            'answer 2': example['ans2'],
        }

        text_to_index_dict = {text: i for i, text in enumerate(answer_list)}

        # For running confusion matrix
        if prediction is None:
            id_result = 3
        elif prediction.lower() in text_to_index_dict:
            id_result = text_to_index_dict[prediction.lower()]
        else:
            id_result = 3
        prediction_list.append(id_result)
        label_list.append(example['label'])

        conf = confusion_matrix(label_list,prediction_list)
        print("confusion_matrix: ",conf)

        acc = accuracy_score(label_list,prediction_list)
        print("accuracy: ",acc)

        # Save results for each batch
        # with open(os.path.join(output_path, output_path.split("/")[-1]+".json"), 'a') as f:
        #     json.dump(results, f, indent=2)
        #     f.write(",")
        #     f.close()
        append_to_json(os.path.join(output_path, output_path.split("/")[-1]+".jsonl"),results)

        # with open(output_path, 'w') as f:
            # json.dump(results, f, indent=2)

    accuracy = correct / total if total > 0 else 0

    print(f"Evaluation complete. Accuracy: {accuracy:.2f}")

    # Save result into a file in the 
    with open(os.path.join(output_path, output_path.split("/")[-1]), "w") as file:
        # Write Confusion matrix
        conf_text = "confusion_matrix: "+str(conf)
        file.write(conf_text)
        acc_text = "accuracy: "+ str(acc)
        file.write(acc_text)
        file.close()
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate NLI model")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset file path")
    args = parser.parse_args()

    dataset = load_evalset(args)
    accuracy = evaluate(args.output, dataset)
    print(f"Final accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()