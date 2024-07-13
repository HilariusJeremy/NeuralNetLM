# A short note here that n in the paper is not the same as the n-gram. But instead n_gram = n - 1
# So given n = 5, the prediction will use 4-gram instead.

import torch
from torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def generate_n_gram(text, n_gram):
        words = word_tokenize(text)
        n_grams = zip(*[words[i:] for i in range(n_gram)])
        n_grams = [" ".join(i) for i in n_grams]
        if n_grams:
            n_grams.pop()
        n_grams = [(n_grams[i], words[i+n_gram]) for i in range(len(n_grams))]
        return list(n_grams)
    
class CustomTextDataset(Dataset):
    def __init__(self, n, df, vocab):
        self.n = n
        self.n_gram = n - 1
        self.vocab = vocab
        self.V = len(vocab)
        temp_df = df['processed_text'].apply(lambda row: generate_n_gram(row, self.n_gram))
        self.data = []
        for _, row in temp_df.items():
            if row:
                self.data += row 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ngram, next_word = self.data[idx]
        ngram_indices = [self.vocab[word] for word in word_tokenize(ngram)]
        next_word_index = self.vocab[next_word]
        ngram_indices_tensor = torch.tensor(ngram_indices).to(device)
        next_word_index_tensor = torch.tensor(next_word_index).to(device)
        return ngram_indices_tensor, next_word_index_tensor

def create_dataloaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        vocab: dict,
        n: int,
        batch_size: int
                        ):
        train_set = CustomTextDataset(n, train_df, vocab)
        val_set = CustomTextDataset(n, val_df, vocab)
        test_set = CustomTextDataset(n, test_df,vocab)

        trainloader = DataLoader(train_set, batch_size=batch_size)
        valloader = DataLoader(val_set, batch_size=batch_size)
        testloader = DataLoader(test_set, batch_size=batch_size)
        return trainloader, valloader, testloader
