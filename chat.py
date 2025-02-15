import torch
from model import ChatModel
from tokenizer import Tokenizer
import os

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long).to(model.head.weight.device)
    
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens.append(next_token.item())
        x = torch.cat([x, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        if len(tokens) >= max_new_tokens:
            break
    
    return tokenizer.decode(tokens)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    
    model = ChatModel(vocab_size=tokenizer.vocab_size).to(device)
    
    if os.path.exists("checkpoints/model.pt"):
        model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
        print("Loaded model from checkpoint")
    else:
        print("No checkpoint found, using untrained model")
    
    print("Chat started. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break
        
        response = generate(model, tokenizer, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()

