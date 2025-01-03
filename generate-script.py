from mlx_lm import load, generate

model, tokenizer = load("qwen2.5-0.5B-ft3000")
system_prompt = "ou are a user input safety monitor of a AI chatbox, and you need to check whether the user's input is safe. Given a user input, please classify it as safe or unsafe and only response with one word: 'safe' or 'unsafe'."

def custom_chat(model, tokenizer, system_prompt):
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

custom_chat(model, tokenizer, system_prompt)