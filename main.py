import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# ================== 超参数设置 ==================
config = {
    'num_epochs': 3,  # 每个任务训练轮数
    'batch_size': 64,
    'lr': 1e-3,  # 学习率
    'lambda': 1000,  # EWC正则化系数
    'fisher_sample': 200,  # 计算Fisher信息的采样数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# ================== 数据准备 ==================
def prepare_tasks():
    """创建两个连续的学习任务（MNIST数字分类）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 任务1：数字0-4
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    task1_indices = [i for i, (_, label) in enumerate(full_dataset) if label < 5]
    task1_dataset = Subset(full_dataset, task1_indices)

    # 任务2：数字5-9
    task2_indices = [i for i, (_, label) in enumerate(full_dataset) if label >= 5]
    task2_dataset = Subset(full_dataset, task2_indices)

    return {
        'task1': DataLoader(task1_dataset, batch_size=config['batch_size'], shuffle=True),
        'task2': DataLoader(task2_dataset, batch_size=config['batch_size'], shuffle=True)
    }


# ================== 模型定义 ==================
class ContinualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        features = self.feature_extractor(x)
        return self.classifier(features)


# ================== EWC核心实现 ==================
class EWC_Regularizer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.means = {}  # 旧任务参数均值
        self.fisher = {}  # Fisher信息矩阵

    def calculate_fisher(self, dataloader):
        """计算Fisher信息矩阵"""
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        # 采样部分数据
        indices = torch.randperm(len(dataloader.dataset))[:config['fisher_sample']]
        subset = Subset(dataloader.dataset, indices)
        temp_loader = DataLoader(subset, batch_size=config['batch_size'])

        # 计算梯度平方期望
        self.model.eval()
        for data, target in temp_loader:
            data, target = data.to(config['device']), target.to(config['device'])
            self.model.zero_grad()
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2).mean(0) * len(data)

        # 平均处理
        for n in fisher:
            fisher[n] /= len(subset)
            self.fisher[n] = fisher[n].cpu()

    def penalty(self, model: nn.Module):
        """计算EWC正则化项"""
        loss = 0
        for n, p in model.named_parameters():
            if n in self.means:
                loss += (self.fisher[n].to(config['device']) *
                         (p - self.means[n].to(config['device'])).pow(2)).sum()
        return config['lambda'] * loss


# ================== 训练流程 ==================
def train_task(model, optimizer, dataloader, ewc=None):
    """单任务训练函数"""
    model.train()
    for epoch in range(config['num_epochs']):
        total_loss = 0
        for data, target in dataloader:
            data, target = data.to(config['device']), target.to(config['device'])

            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)

            # 添加EWC正则项
            if ewc is not None:
                loss += ewc.penalty(model)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}')


def evaluate(model, dataloader):
    """模型评估函数"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(config['device']), target.to(config['device'])
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(dataloader.dataset)
    return accuracy


# ================== 主流程 ==================
if __name__ == "__main__":
    # 初始化模型和优化器
    model = ContinualModel().to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 准备任务数据
    loaders = prepare_tasks()

    # 阶段1：训练任务1 (数字0-4)
    print("==== Training Task 1 ====")
    train_task(model, optimizer, loaders['task1'])
    task1_acc = evaluate(model, loaders['task1'])
    print(f"Task1 Accuracy: {task1_acc:.2f}%")

    # 计算Fisher信息并保存参数
    ewc = EWC_Regularizer(model)
    ewc.calculate_fisher(loaders['task1'])
    ewc.means = {n: p.clone().detach().cpu() for n, p in model.named_parameters()}

    # 阶段2：训练任务2 (数字5-9) 应用EWC
    print("\n==== Training Task 2 with EWC ====")
    train_task(model, optimizer, loaders['task2'], ewc=ewc)

    # 最终评估
    print("\n==== Final Evaluation ====")
    task1_final_acc = evaluate(model, loaders['task1'])
    task2_acc = evaluate(model, loaders['task2'])
    print(f"Task1 Forget Rate: {task1_acc - task1_final_acc:.2f}%")
    print(f"Task2 Accuracy: {task2_acc:.2f}%")