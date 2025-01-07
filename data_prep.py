import os
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--format", "-f", required=False, help="data format of the processed data", default="chat", choices=["chat", "completion"])
args = parser.parse_args()

# load data
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
train_df = pd.read_parquet("hf://datasets/xTRam1/safe-guard-prompt-injection/" + splits["train"])
test_df =  pd.read_parquet("hf://datasets/xTRam1/safe-guard-prompt-injection/" + splits["test"])


def filter_data(data):
    ### avoid long sentence
    filtered_data = []
    for item in data:
        if len(item['text']) <= 2048: # change according to your data format, maybe 'prompt'
            filtered_data.append(item)
    return filtered_data

def convert_chat(data):
    label_mapping = {0: "safe", 1: "unsafe"}
    formatted_data = []
    for item in data:
        messages = [
            {"role": "system", "content": "You are a user input safety monitor of a AI chatbox, and you need to check whether the user's input is safe. Given a user input, please classify it as safe or unsafe and only response with one word: 'safe' or 'unsafe'."},
            {"role": "user", "content": item["text"]},
            {"role": "assistant", "content": label_mapping[item["label"]]} # if test data don't have "label", remove this
        ]
        formatted_data.append({"messages": messages})
    return formatted_data

def convert_completion(data):
    label_mapping = {0: "safe", 1: "unsafe"}
    modified_data = []
    for item in data:
        new_item = {
        "prompt": item['text'],
        "completion": label_mapping[item["label"]]
        }
    modified_data.append(new_item)
    return modified_data

def process_and_save(data, filename):
    filtered_data = filter_data(data)
    if args.format == "chat":
        converted_data = convert_chat(filtered_data)
    else:  # "completion" mode
        converted_data = convert_completion(filtered_data)
        
    output_dir = f"data/{args.format}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')

def main():
    train_records = filter_data(train_df.to_dict(orient='records'))
    train_data = train_records[:7000]
    valid_data = train_records[7000:]
    test_data = filter_data(test_df.to_dict(orient='records'))
    
    process_and_save(train_data, "train.jsonl")
    process_and_save(valid_data, "valid.jsonl")
    process_and_save(test_data, "test.jsonl")
    
if __name__ == "__main__":
    main()
