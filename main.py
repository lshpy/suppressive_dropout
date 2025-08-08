import os, argparse, importlib, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import optim
from model.cnn import get_cnn
from experiments.suppressive_dropout import SuppressiveDropout
from train import set_seed, train_one_epoch
from evaluate import evaluate
from utils.logger import save_json, save_logs_csv

def get_loaders(batch_size=128, val_ratio=0.2):
    tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465],[0.247,0.243,0.261])
    ])
    tfm_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465],[0.247,0.243,0.261])
    ])
    root='./data'
    train_full = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
    test_set   = datasets.CIFAR10(root, train=False, download=True, transform=tfm_t)
    n_train = int((1.0 - val_ratio) * len(train_full))
    n_val   = len(train_full) - n_train
    train_set, val_set = random_split(train_full, [n_train, n_val])
    args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    return (DataLoader(train_set, shuffle=True, **args),
            DataLoader(val_set,   shuffle=False, **args),
            DataLoader(test_set,  shuffle=False, **args))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_ratio", type=float, default=0.2)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, default="results/cnn_sdrop")
    ap.add_argument("--use_sdrop", action="store_true", help="켜면 Suppressive Dropout 적용")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(args.batch_size)

    drop_layer = SuppressiveDropout(args.drop_ratio, b=args.b, c=args.c) if args.use_sdrop else None
    model = get_cnn(num_classes=10, drop_layer=drop_layer).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)

    logs = []
    for epoch in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, device, opt, sch)
        val = evaluate(model, val_loader, device)
        log = dict(epoch=epoch, **{f"train_{k}": v for k,v in tr.items()}, **{f"val_{k}": v for k,v in val.items()})
        print(f"[Epoch {epoch}] "
              f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.2f} | "
              f"val_acc={val['acc']:.2f} f1M={val['f1_macro']:.3f} auc={val['auc']:.3f} ece={val['ece']:.4f}")
        logs.append(log)

    # 최종 테스트
    test = evaluate(model, test_loader, device)
    print(f"[TEST] acc={test['acc']:.2f} f1M={test['f1_macro']:.3f} auc={test['auc']:.3f} ece={test['ece']:.4f}")

    # 저장
    save_logs_csv(logs, os.path.join(args.save_dir, f"r{args.drop_ratio:.2f}_b{args.b}_c{args.c}"))
    save_json(test, os.path.join(args.save_dir, f"summary_r{args.drop_ratio:.2f}_b{args.b}_c{args.c}.json"))
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints",
                                                f"cnn_sdrop_r{args.drop_ratio:.2f}_b{args.b}_c{args.c}.pt"))

if __name__ == "__main__":
    main()
