# from mlx_lm import load, generate
# import json

# # model, tokenizer = load('Qwen/Qwen2-0.5B-Instruct-MLX', tokenizer_config={"eos_token": "<|im_end|>"})
# model, tokenizer = load('qwen2-0.5b-v3', tokenizer_config={"eos_token": "<|im_end|>"})

# prompts = []
# with open("data/test-v2.jsonl", "r") as file:
#     for line in file:
#         data = json.loads(line)
#         prompts.append(data["prompt"])

# messages_list = [
#     [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
#     for prompt in prompts
# ]

# texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]

# responses = []
# for text in texts:
#     response = generate(model, tokenizer, prompt=text, verbose=True, top_p=0.8, temp=0.7, repetition_penalty=1.05, max_tokens=512)
#     print(response)
#     responses.append(response)

# with open("responses-train-v3.jsonl", "w") as file:
#     for prompt, response in zip(prompts, responses):
#         data = {"prompt": prompt, "response": response}
#         file.write(json.dumps(data) + "\n")

from mlx_lm import load, generate
import json

model, tokenizer = load('qwen2.5-0.5B-ft3000', tokenizer_config={"eos_token": "<|im_end|>"})

input_file = 'test.jsonl'
output_file = 'response-testnew.jsonl'

with open(input_file, 'r', encoding='utf-8') as f:
    messages_list = [json.loads(line) for line in f]

generated_responses = []

for messages in messages_list:
    text = tokenizer.apply_chat_template(
        messages['messages'],
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(model, tokenizer, prompt=text, verbose=True, top_p=0.8, temp=0.7, repetition_penalty=1.05, max_tokens=512)
    generated_responses.append(response)

with open(output_file, 'w', encoding='utf-8') as f:
    for response in generated_responses:
        f.write(json.dumps({"response": response}) + '\n')