import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List
import tqdm


def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):
    if mode=='train':
        model.train()
    elif mode=='test':
        model.eval()
    cost = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode=='train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
    cost = cost / len(dataset)
    return cost


def train(model: torch.nn.Module,
          train_set: torch.utils.data.Dataset,
          test_set: torch.utils.data.Dataset,
          criterion: torch.nn.Module,
          trainloader: torch.utils.data.DataLoader,
          testloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    epochs = epochs
    train_cost, test_cost = [], []
    for i in range(epochs):
      cost = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    train_cost.append(cost)
    with torch.no_grad():
        cost = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)
        test_cost.append(cost)
    
    print(f"\rEpoch: {i+1}/{epochs} | train_cost: {train_cost[-1]: 4f} | test_cost: {test_cost[-1]: 4f} | ", end=" ")
  