import os
import argparse
import json
from ollama import Client

parser = argparse.ArgumentParser(description="Load and generate text using a specified model.")
parser.add_argument("--input", "-i", required=True, help="File to predict.")
parser.add_argument("--output", "-o", required=True, help="Save the prediction to specific path.")
    
args = parser.parse_args()

def main():
    client = Client()

    input_data = []
    with open(args.input, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'input' in data:
                    input_data.append(data['input'])
            except json.JSONDecodeError:
                print(f"invalid line: {line}")

    all_predictions = []
    
    for input_text in input_data:
        response = client.generate(model='gguf-example:latest', prompt=input_text)
        prediction = {
            'input': input_text,
            'output': response.response
        }
        all_predictions.append(prediction)
        
    with open(args.output, 'w') as f:
        json.dump(all_predictions, f, indent=4)
    print("Finished.")

if __name__ == "__main__":
    main()