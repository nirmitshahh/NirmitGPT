import tiktoken

class Tokenizer:
    def __init__(self, encoding_name="cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, tokens):
        return self.enc.decode(tokens)
    
    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists):
        return [self.decode(tokens) for tokens in token_lists]

