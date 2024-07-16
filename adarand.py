import datetime
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision
from models import CustomResNet50, RandomFeatureSampler, adarand_loss
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
    reg_lambda: float = 1.0
    rand_lr: float = 1e-3


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

    resnet50 = CustomResNet50(num_classes=len(test_dataset.classes))
    resnet50 = torch.nn.DataParallel(resnet50)
    resnet50.to(device)

    precomputed_sum1 = torch.zeros(len(test_dataset.classes), resnet50.module.resnet50.fc.in_features).to(device)
    precomputed_sum2 = torch.zeros(len(test_dataset.classes), resnet50.module.resnet50.fc.in_features).to(device)
    class_nums = torch.zeros(len(test_dataset.classes)).to(device)
    with torch.no_grad():
        for x, y in tqdm(train_loader, total=len(train_loader), desc="Precomputing mu and sigma"):
            x, y = x.to(device), y.to(device)
            _, feat = resnet50(x)
            y_onehot = torch.nn.functional.one_hot(y, len(test_dataset.classes)).float()
            precomputed_sum1 += y_onehot.T @ feat
            precomputed_sum2 += y_onehot.T @ (feat**2)
            class_nums += y_onehot.sum(dim=0)

    mu = precomputed_sum1 / class_nums.view(-1, 1)
    sigma = torch.sqrt((precomputed_sum2 / class_nums.view(-1, 1)) - mu**2)

    random_features = RandomFeatureSampler(
        feat_dim=resnet50.module.resnet50.fc.in_features,
        num_classes=len(test_dataset.classes),
        mu=mu,
        sigma=sigma,
    )
    random_features.to(device)

    model_optimizer = torch.optim.AdamW(resnet50.parameters(), lr=config.model_lr, weight_decay=1e-4)
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer,
        milestones=config.model_lr_milestones,
        gamma=config.model_lr_drop_rate,
    )

    rand_optimizer = torch.optim.Adam(random_features.parameters(), lr=config.rand_lr)
    rand_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rand_optimizer, T_max=config.max_iter)

    best_val_acc = 0.0
    best_state_dict = None
    for epoch in range(config.max_iter):
        resnet50.train()
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_reg = 0.0
        train_size = 0
        for x, y in tqdm(train_loader, total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            model_optimizer.zero_grad()

            y_pred, feat = resnet50(x)
            loss_ce = torch.nn.functional.cross_entropy(y_pred, y)
            loss_reg = torch.nn.functional.mse_loss(random_features(y), feat)
            loss = loss_ce + config.reg_lambda * loss_reg
            loss.backward()
            model_optimizer.step()

            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_reg += loss_reg.item()
            train_size += len(y)

            rand_optimizer.zero_grad()
            rand_loss = adarand_loss(random_features, feat.clone().detach(), y)
            rand_loss.backward()
            rand_optimizer.step()

        model_scheduler.step()
        rand_scheduler.step()

        resnet50.eval()
        top1_correct = 0
        top5_correct = 0
        val_size = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred, _ = resnet50(x)
                _, predicted = torch.max(y_pred, 1)
                val_size += y.size(0)
                top1_correct += (predicted == y).sum().item()
                _, predicted_top5 = torch.topk(y_pred, 5, dim=1)
                for i in range(y.size(0)):
                    top5_correct += y[i] in predicted_top5[i]

        top1_acc = top1_correct / val_size
        top5_acc = top5_correct / val_size

        epoch_result = f"Epoch {epoch}/{config.max_iter}, Loss: {running_loss / train_size}, CE: {running_loss_ce / train_size}, Reg: {running_loss_reg / train_size}, Top-1 Acc: {top1_acc}, Top-5 Acc: {top5_acc}"
        print(epoch_result)
        with open(log_path, "a") as f:
            f.write(epoch_result + "\n")

        if top1_acc > best_val_acc:
            best_val_acc = top1_acc
            best_state_dict = deepcopy(resnet50.state_dict())

    if best_state_dict:
        torch.save(best_state_dict, outdir / f"best_state_dict.pth")
        resnet50.load_state_dict(best_state_dict)
    resnet50.eval()
    top1_correct = 0
    top5_correct = 0
    test_size = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred, _ = resnet50(x)
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
