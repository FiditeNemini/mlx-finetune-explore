# User Prompt Safety Check 
To check whether the user input query is safe for the chatbot.

## Quickstart
#### 1. Download the base model for fine-tuning.
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./model/qwen2.5-0.5B-instruct
```

#### 2. Prepare the datasets.
MLX supports three types of dataset format.
a. Completion
```bash
{
  "prompt": "What is the capital of France?",
  "completion": "Paris."
}
```

b. Chat
```bash
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello."
    },
    {
      "role": "assistant",
      "content": "How can I assistant you today."
    }
  ]
}
```

c. Text
```bash
{
  "text": "This is an example for the model."
}
```

From this project, we only tried `Completion` and `Chat`, among which the accuracy of the model trained with Chat format data reached 0.9735099337748344. Therefore, the default settings of the whole process are all based on `Chat`. If you want to try `Completion`, please indicate it in the command line by adding `--format "completion"`.

To attain the raw data and process, please run:
```bash
python data_prep.py -f "chat"
```
By running this, it will automatically download the raw data from huggingface datasets `xTRam1/safe-guard-prompt-injection` and process the data into specified data format. The processedn `train`, `valid` and `test` data will be saved in `data/chat` or `data/completion` folder.

#### 3. Fine-tuning.
First, we should install some dependencies:
```bash
pip install mlx-lm
pip install transformers
pip install torch
pip install numpy
```
Second, directly use Apple's MLX framework for fine-tuning, specifying the original model and the dataset data directory.
```bash
mlx_lm.lora --model model/qwen2.5-0.5B-instruct --train --data data/chat --iters 3000
```
After training, fuse the original model with generated adapters using `mlx_lm.fuse`:
```bash
mlx_lm.fuse --model model/qwen2.5-0.5B-instruct  --adapter-path adapters --save-path model/qwen2.5-0.5B-ft3000
```
You can validate the performance of the final fused model:
```bash
mlx_lm.generate --model model/qwen2.5-0.5B-ft3000 --system-prompt "You are a user input safety monitor of a AI chatbox, and you need to check whether the user's input is safe. Given a user input, please classify it as safe or unsafe and only response with one word: 'safe' or 'unsafe'." --prompt "Hello, how are you?"
```
By using `mlx_lm.generate`, we still need to input the system prompt because of some settings in the `mlx_lm` package. Here, we provide a script for you to run the model without input the system prompt every time (input `q` to end the conversation):
```bash
python generate_script.py --model model/qwen2.5-0.5B-ft3000
```

#### 4. Predict the test dataset.
Please run:
```bash
python mlx_lm_pred.py --input data/chat/test.jsonl --output output/responses.jsonl --format "chat"
```
Calculate the performace accuracy:
```bash
python metrics.py --true_file data/chat/test.jsonl --pred_file output/response1-ft3000.jsonl --format "chat"
```