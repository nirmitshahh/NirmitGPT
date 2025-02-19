import torch
import os
import json

def save_model(model, path, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, path)

def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('config', None)

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)

