import torch
import torch.nn as nn
from torch.optim import AdamW
from model import ChatModel
from tokenizer import Tokenizer
from data import get_dataloader
import os

def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} avg loss: {total_loss / len(dataloader):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    model = ChatModel(vocab_size=vocab_size).to(device)
    
    # dummy data for now
    texts = ["Hello, how are you?", "I'm doing well, thanks!"]
    dataloader = get_dataloader(texts, tokenizer, batch_size=2)
    
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    train(model, dataloader, optimizer, device, epochs=5)
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")
    print("Model saved")

if __name__ == "__main__":
    main()

