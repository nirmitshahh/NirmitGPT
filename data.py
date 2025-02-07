import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer

class ChatDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def get_dataloader(texts, tokenizer, batch_size=32, max_length=1024, shuffle=True):
    dataset = ChatDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    
    x_padded = []
    y_padded = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        x_padded.append(torch.cat([x, torch.zeros(pad_len, dtype=torch.long)]))
        y_padded.append(torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)]))
    
    return torch.stack(x_padded), torch.stack(y_padded)

