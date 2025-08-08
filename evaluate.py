import torch
from utils.metrics import evaluate_np

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(dim=1)
        all_probs.append(probs.cpu()); all_preds.append(preds.cpu()); all_labels.append(y.cpu())
    import torch as T
    labels_np = T.cat(all_labels).numpy()
    preds_np  = T.cat(all_preds).numpy()
    probs_np  = T.cat(all_probs).numpy()
    return evaluate_np(labels_np, preds_np, probs_np)
