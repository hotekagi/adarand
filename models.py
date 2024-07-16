import torch
import torchvision
from torch import nn
from torchvision.models.resnet import ResNet50_Weights


class RandomFeatureSampler(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        mu: int | float | torch.Tensor,
        sigma: int | float | torch.Tensor,
        alpha: float = 0.5,
    ) -> None:
        """
        Args:
            feat_dim: Dimension of the feature vector.
            num_classes: Number of classes.
            mu: Mean of the normal distribution. If int or float, it will be broadcasted to (num_classes, feat_dim).
            sigma: Standard deviation of the normal distribution. (diagnoal part only)
                If int or float, it will be broadcasted to (num_classes, feat_dim).
            alpha: Exponential moving average decay.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

        if isinstance(mu, float) or isinstance(mu, int):
            self.mu = torch.nn.Parameter(
                torch.tensor([mu] * feat_dim * num_classes).float().view(-1, feat_dim),
                requires_grad=True,
            )
        else:
            self.mu = torch.nn.Parameter(mu.clone().detach(), requires_grad=True)

        if isinstance(sigma, float) or isinstance(sigma, int):
            self.sigma = torch.nn.Parameter(
                torch.tensor([sigma] * feat_dim * num_classes).float().view(-1, feat_dim),
                requires_grad=True,
            )
        else:
            sigma = torch.clamp(sigma, min=1e-6)
            self.sigma = torch.nn.Parameter(sigma.clone().detach(), requires_grad=True)

        if (
            self.mu.size()[0] != num_classes
            or self.mu.size()[1] != feat_dim
            or self.sigma.size()[0] != num_classes
            or self.sigma.size()[1] != feat_dim
        ):
            raise ValueError("mu and sigma must have the same shape as (num_class, feat_dim)")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        batch_mu = self.mu[y]
        batch_sigma = self.sigma[y]
        return torch.distributions.Normal(batch_mu, batch_sigma).sample()

    def ema_mu(self, batch_mu: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mu.data.clone().detach() + (1 - self.alpha) * batch_mu


def adarand_loss(dist: RandomFeatureSampler, batch_feat: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
    batch_y_onehot = torch.nn.functional.one_hot(batch_y, dist.num_classes).float()
    batch_mu = batch_y_onehot.T @ batch_feat
    batch_count = batch_y_onehot.sum(dim=0)
    batch_mu[batch_count > 0] = batch_mu[batch_count > 0] / batch_count[batch_count > 0].view(-1, 1)
    ema_mu = dist.ema_mu(batch_mu)

    loss_intra = (1.0 - torch.nn.functional.cosine_similarity(dist.mu, ema_mu, dim=1)).mean()
    sim_matrix = torch.nn.functional.cosine_similarity(dist.mu.unsqueeze(1), dist.mu.unsqueeze(0), dim=2)
    mask = 1.0 - torch.eye(dist.num_classes, device=dist.mu.device)
    loss_inter = (sim_matrix * mask).mean()

    return loss_intra + loss_inter


class CustomResNet50(torch.nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.fc = torch.nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)

        features = x.clone()

        x = self.resnet50.fc(x)

        return x, features
