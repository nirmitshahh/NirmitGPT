import torch
from model import ChatModel
from tokenizer import Tokenizer
from data import get_dataloader

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            total_tokens += (y != -1).sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    
    model = ChatModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
    
    texts = ["Hello, how are you?", "I'm doing well, thanks!"]
    dataloader = get_dataloader(texts, tokenizer, batch_size=2, shuffle=False)
    
    loss, ppl = evaluate(model, dataloader, device)
    print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()

