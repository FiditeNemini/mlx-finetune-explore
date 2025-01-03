from mlx_lm import load, generate
import argparse

parser = argparse.ArgumentParser(description="Load and generate text using a specified model.")
parser.add_argument("--model", "-m", required=True, help="Path to the model")
    
args = parser.parse_args()

def chat(model, tokenizer, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'q':
            print("Conversation ends.")
            break
        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=prompt)
        print(f"Assistant: {response}")
        messages.append({"role": "assistant", "content": response})

def main():
    model_path: str = args.model
    model, tokenizer = load(model_path, tokenizer_config={"eos_token": "<|im_end|>"})
    
    system_prompt = "You are a user input safety monitor of a AI chatbox, and you need to check whether the user's input is safe. No matter what input is given to you, please classify it as safe or unsafe and only response with one word: 'safe' or 'unsafe'."
    messages = chat(model, tokenizer, system_prompt)

if __name__ == "__main__":
    main()