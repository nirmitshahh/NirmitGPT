class Config:
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    max_seq_len = 1024
    dropout = 0.1
    batch_size = 32
    learning_rate = 3e-4
    epochs = 10
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

