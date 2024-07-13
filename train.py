import os
import torch
import data_setup, model_builder, engine
import pandas as pd
import re
from nltk import word_tokenize
from collections import Counter
from torch import nn, optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 20
BATCH_SIZE = 4096
H = 50
N = 5
M = 60
LEARNING_RATE = 0.001

def alphanumeric_cleansing(text):
    # Remove non-alphanumeric characters using regular expression
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

if __name__ == "__main__":
    df = pd.read_csv('./brown.csv')
    df['tokenized_text'] = df['tokenized_text'].apply(lambda row: alphanumeric_cleansing(row))
    df['tokens'] = df['tokenized_text'].apply(word_tokenize)
    # Merge rare words
    all_tokens = [token for sublist in df['tokens'] for token in sublist]
    token_freq = Counter(all_tokens)
    df['tokens'] = df['tokens'].apply(lambda tokens: ['UNKNOWN' if token_freq[token]<=3 else token for token in tokens])
    df['processed_text'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
    tokens = {token for sublist in df['tokens'] for token in sublist}
    sorted_tokens = sorted(tokens)
    vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
    token_to_word = {v: k for k, v in vocab.items()}
    counter = 0
    flag = 0
    train_sep = -1
    val_sep = -1
    for index, row in df.iterrows():
      counter += len(row.tokens)
      if counter >= 800000 and flag == 0:
          train_sep = index
          counter = 0
          flag = 1
          continue
      if counter >= 200000 and flag == 1:
          val_sep = index
          break

    train_df = df.iloc[:train_sep] 
    val_df = df.iloc[train_sep+1:val_sep]
    test_df = df.iloc[val_sep+1:]
    train_set, val_set, test_set, trainloader, valloader, testloader = data_setup.create_dataloaders(train_df, val_df, test_df, vocab, N, BATCH_SIZE)
    model = model_builder.NeuralNetLM(vocab, m = M, n = N, h = H).to(device)
    criterion = nn.NLLLoss() # Training is actually done with 2 loss functions, weights use L2, 
                         # Biases use max. likelihood without regularization term.
    optimizer = optim.AdamW(model.parameters())
    engine.train(model, train_set, val_set, criterion, trainloader, valloader, optimizer, NUM_EPOCHS, device)
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/weights.pth')








