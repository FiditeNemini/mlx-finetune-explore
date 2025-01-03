import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, help="path to the input data")
parser.add_argument("--output", "-o", required=True, help="path to the output file")
parser.add_argument("--format", "-f", required=False, help="data format of the processed data", default="chat", choices=["chat", "completion"])

args = parser.parse_args()

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
    
def main():
    output_dir: str = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    input_data: list
    with open(args.input, "r") as in_file:
        input_data = json.load(in_file)
    
    filtered_data: list = filter_data(input_data)
    if args.format == "chat":
        converted_data: list = convert_chat(filtered_data)
    else: # "completion" mode
        converted_data: list = convert_completion(filtered_data)

    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    main()
