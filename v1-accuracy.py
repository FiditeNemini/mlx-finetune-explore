# %% [markdown]
# ## Raw Accuracy

# %%
import json

# Function to calculate accuracy
def calculate_accuracy():
    import json
    true_data = []
    pred_data = []

    # Load true data
    with open('data/filtered_test_data.jsonl', 'r') as f:
        for line in f:
            true_data.append(json.loads(line))

    # Load predicted data
    with open('responses-0.5B.jsonl', 'r') as f:
        for line in f:
            pred_data.append(json.loads(line))

    correct_predictions = 0
    false_predictions = 0
    count_text = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Compare completion and response fields
    for true_entry, pred_entry in zip(true_data, pred_data):
        if true_entry['prompt'] == pred_entry['prompt']:
            count_text += 1
        if true_entry['completion'] == pred_entry['response']:
            correct_predictions += 1
            if true_entry['completion']=='safe' and pred_entry['response']=='safe':
                true_negative += 1
            else:
                true_positive += 1
        elif true_entry['completion']=='safe' and pred_entry['response']=='unsafe':
            false_predictions += 1
            false_positive += 1
        elif true_entry['completion']=='unsafe' and pred_entry['response']=='safe':
            false_predictions += 1
            false_negative += 1
        else:
            false_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(true_data)
    missing_rate = false_predictions / len(true_data)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2*precision*recall/(precision+recall)
    return accuracy, missing_rate, precision, recall, f1

# Calculate and print the accuracy
accuracy_05B = calculate_accuracy()
print(accuracy_05B)

# %% [markdown]
# ## Fine-tuned Accuracy 

# %%
import json

# Function to calculate accuracy
def calculate_accuracy():
    import json
    true_data = []
    pred_data = []

    # Load true data
    with open('comp.jsonl', 'r') as f:
        for line in f:
            true_data.append(json.loads(line))

    # Load predicted data
    # with open('results.json', 'r') as f:
    #     pred_data = json.load(f)
    with open('response-testnew.jsonl', 'r') as f:
        for line in f:
            pred_data.append(json.loads(line))

    # correct_predictions = 0
    # count_text = 0

    # # completion mode
    # # Compare completion and response fields
    # for true_entry, pred_entry in zip(true_data, pred_data):
    #     if true_entry['completion'] == pred_entry['response']:
    #         correct_predictions += 1

    # # chat mode
    # # for true_entry, pred_entry in zip(true_data, pred_data):
    # #     if true_entry['completion'] == pred_entry['response']:
    # #         correct_predictions += 1

    # # Calculate accuracy
    # accuracy = correct_predictions / len(true_data)

    # return accuracy, count_text
    correct_predictions = 0
    false_predictions = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Compare completion and response fields
    for true_entry, pred_entry in zip(true_data, pred_data):
        if true_entry['prompt_label'] == pred_entry['response']:
            correct_predictions += 1
            if true_entry['prompt_label'] == 'safe' and pred_entry[
                    'response'] == 'safe':
                true_negative += 1
            else:
                true_positive += 1
        elif true_entry['prompt_label'] == 'safe' and pred_entry[
                'response'] == 'unsafe':
            false_predictions += 1
            false_positive += 1
        elif true_entry['prompt_label'] == 'unsafe' and pred_entry[
                'response'] == 'safe':
            false_predictions += 1
            false_negative += 1
        else:
            false_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(true_data)
    missing_rate = false_predictions / len(true_data)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, missing_rate, precision, recall, f1

# Calculate and print the accuracy
accuracy_train = calculate_accuracy()
print(accuracy_train)
## accuracy without finetune: 0.7972491085073866
## accuracy with finetune: 0.9551706571574121


