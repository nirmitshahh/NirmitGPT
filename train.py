import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import ChatModel
from tokenizer import Tokenizer
from data import get_dataloader
import os

def train(model, dataloader, optimizer, scheduler, device, epochs=10, save_every=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        
        if scheduler:
            scheduler.step()
        
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    model = ChatModel(vocab_size=vocab_size).to(device)
    
    # load data from file if exists, otherwise use dummy
    try:
        with open("data.txt", "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        texts = ["Hello, how are you?", "I'm doing well, thanks!"]
    
    dataloader = get_dataloader(texts, tokenizer, batch_size=2)
    
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    
    train(model, dataloader, optimizer, scheduler, device, epochs=5)
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")
    print("Model saved")

if __name__ == "__main__":
    main()

