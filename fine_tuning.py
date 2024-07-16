import datetime
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


@dataclass
class Config:
    seed: int = 42
    val_ratio: float = 0.2
    data_path: Path = Path(__file__).parent
    batch_size: int = 32
    max_iter: int = 200
    model_lr: float = 1e-3
    model_lr_milestones: tuple[int, ...] = (60, 120, 160)
    model_lr_drop_rate: float = 0.1


def run_exp(config: Config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda:0" if (torch.cuda.is_available()) else "cpu"

    outdir = Path(__file__).parent / "results" / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "log.txt"
    print(f"Logging to {log_path}")
    with open(log_path, "w") as f:
        f.write(f"Executing {__file__}\n")
        f.write(f"Config: {asdict(config)}\n")

    trainval_data = torchvision.datasets.StanfordCars(
        root=config.data_path,
        split="train",
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
        ),
    )
    train_dataset, val_dataset = random_split(trainval_data, [1 - config.val_ratio, config.val_ratio])
    test_dataset = torchvision.datasets.StanfordCars(
        root=config.data_path,
        split="test",
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
        ),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, len(test_dataset.classes))

    resnet50 = torch.nn.DataParallel(resnet50)
    resnet50.to(device)

    model_optimizer = torch.optim.AdamW(resnet50.parameters(), lr=config.model_lr, weight_decay=1e-4)
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer,
        milestones=config.model_lr_milestones,
        gamma=config.model_lr_drop_rate,
    )

    best_val_acc = 0.0
    best_state_dict = None
    for epoch in range(config.max_iter):
        resnet50.train()
        running_loss = 0.0
        train_size = 0
        for x, y in tqdm(train_loader, total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            model_optimizer.zero_grad()

            y_pred = resnet50(x)
            loss = torch.nn.functional.cross_entropy(y_pred, y)
            loss.backward()
            model_optimizer.step()

            running_loss += loss.item()
            train_size += len(y)

        model_scheduler.step()

        resnet50.eval()
        top1_correct = 0
        top5_correct = 0
        val_size = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = resnet50(x)
                _, predicted = torch.max(y_pred, 1)
                val_size += y.size(0)
                top1_correct += (predicted == y).sum().item()
                _, predicted_top5 = torch.topk(y_pred, 5, dim=1)
                for i in range(y.size(0)):
                    top5_correct += y[i] in predicted_top5[i]

        top1_acc = top1_correct / val_size
        top5_acc = top5_correct / val_size

        epoch_result = f"Epoch {epoch}/{config.max_iter}, Loss: {running_loss / train_size}, Top-1 Acc: {top1_acc}, Top-5 Acc: {top5_acc}"
        print(epoch_result)
        with open(log_path, "a") as f:
            f.write(epoch_result + "\n")

        if top1_acc > best_val_acc:
            best_val_acc = top1_acc
            best_state_dict = deepcopy(resnet50.state_dict())

    if best_state_dict:
        torch.save(best_state_dict, outdir / "best_state_dict.pth")
        resnet50.load_state_dict(best_state_dict)
    resnet50.eval()
    top1_correct = 0
    top5_correct = 0
    test_size = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = resnet50(x)
            _, predicted = torch.max(y_pred, 1)
            test_size += y.size(0)
            top1_correct += (predicted == y).sum().item()
            _, predicted_top5 = torch.topk(y_pred, 5, dim=1)
            for i in range(y.size(0)):
                top5_correct += y[i] in predicted_top5[i]

    top1_acc = top1_correct / test_size
    top5_acc = top5_correct / test_size
    print(f"Test Top-1 Acc: {top1_acc}, Top-5 Acc: {top5_acc}")
    with open(log_path, "a") as f:
        f.write(f"Test Top-1 Acc: {top1_acc}, Top-5 Acc: {top5_acc}\n")


if __name__ == "__main__":
    config = Config()
    run_exp(config)
