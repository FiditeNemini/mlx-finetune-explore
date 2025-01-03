# %% [markdown]
# 修改數據格式，以便進行訓練

# %%
### Step 1: formatting the raw data ###
import json

# Read the raw data from train.json
# with open('safety_train_data.json', 'r') as file:
with open('safety_test_data.json', 'r') as file:
    json_objects = [line.strip() for line in file if line.strip()]

# Parse each JSON object and add it to a list
json_list = [json.loads(obj) for obj in json_objects]

# Convert the list of JSON objects to a JSON array
correct_json_format = json.dumps(json_list, indent=2)

# Write the corrected JSON format to a new file
with open('test_corrected.json', 'w') as file:
    file.write(correct_json_format)

print("The data has been successfully converted and stored.")

# %%
### step2: prepare the train/test data with correct prompt ###
import json

def filter_data(data):
    filtered_data = []
    for item in data:
        if len(item['text']) <= 2048:
            filtered_data.append(item)
    return filtered_data

# Load the JSON dataset
with open('raw/train_corrected.json', 'r') as f:
# with open('raw/test_corrected.json', 'r') as f:
    data = json.load(f)

data = filter_data(data)
label_mapping = {0: "safe", 1: "unsafe"}
# Create a new list to store the modified data
modified_data = []

# Iterate over each item in the dataset
for item in data:
    new_item = {
        "prompt": item['text'],
        "completion": label_mapping[item["label"]]
    }
    modified_data.append(new_item)

# Write the modified data to a new JSONL file

with open('train-v2.jsonl', 'w') as f:
# with open('test-v2.jsonl', 'w') as f:
    for item in modified_data:
        f.write(json.dumps(item) + '\n')

print("The data has been successfully modified and saved.")

# %% [markdown]
# ## 第二種處理

# %%
import json

# 將輸入數據轉換為指定格式的函數，並將label轉換成對應的文字
def filter_data(data):
    filtered_data = []
    for item in data:
        if len(item['prompt']) <= 2048:
            filtered_data.append(item)
    return filtered_data

def convert_format(data):
    formatted_data = []
    for item in data:
        messages = [
            {"role": "system", "content": "You are a user input safety monitor of a AI chatbox, and you need to check whether the user's input is safe. Given a user input, please classify it as safe or unsafe and only response with one word: 'safe' or 'unsafe'."},
            {"role": "user", "content": item["prompt"]}
            # {"role": "assistant", "content": label_mapping[item["label"]]}
        ]
        formatted_data.append({"messages": messages})
    return formatted_data

input_data = []

# 讀取輸入JSON文件
# with open('raw/train_corrected.json', 'r') as f:
# with open('raw/test_corrected.json', 'r') as f:
    # input_data = json.load(f)
with open('new-test.jsonl', 'r') as f:
    for line in f:
        input_data.append(json.loads(line))


filtered_data = filter_data(input_data)
converted_data = convert_format(filtered_data)

# Write the converted data to a JSONL file
# with open('train-v1.jsonl', 'w', encoding='utf-8') as f:
with open('test-new-1.jsonl', 'w', encoding='utf-8') as f:
    for entry in converted_data:
        f.write(json.dumps(entry) + '\n')

print("finished.")


# %%



