import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, device, optimizer, scheduler=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, running_loss = 0, 0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    if scheduler is not None:
        scheduler.step()

    return dict(loss=running_loss/len(loader), acc=100.0*correct/total)
