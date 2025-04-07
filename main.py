import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ========== 网络结构 ==========
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# ========== EWC 实现 ==========
class EWC:
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                loss += _loss.sum()
        return loss

# ========== GAN ==========
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_gan(generator, discriminator, dataloader, device='cpu', epochs=5):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0, 0
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # 训练判别器
            z = torch.randn(batch_size, 100).to(device)
            fake_data = generator(z).detach()
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data)
            d_loss = criterion(d_real, real_label) + criterion(d_fake, fake_label)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            z = torch.randn(batch_size, 100).to(device)
            fake_data = generator(z)
            d_fake = discriminator(fake_data)
            g_loss = criterion(d_fake, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()

        print(f"GAN Epoch {epoch+1} Loss: {g_loss_total:.4f}")

# ========== FGSM 攻击 ==========
class FGSM:
    def __init__(self, model, epsilon=0.2):
        self.model = model
        self.epsilon = epsilon

    def generate(self, x, y):
        x_adv = x.clone().detach().requires_grad_(True)
        output = self.model(x_adv)
        loss = nn.functional.cross_entropy(output, y)
        self.model.zero_grad()
        loss.backward()
        x_adv = x_adv + self.epsilon * x_adv.grad.sign()
        return x_adv.detach()

# ========== 训练与评估函数 ==========
def train_task(model, optimizer, dataloader, ewc=None, lambda_ewc=1000, gan=None, use_gan=False, device='cpu'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        total_loss = 0
        for real_data, real_target in dataloader:
            real_data, real_target = real_data.to(device), real_target.to(device)
            if use_gan and gan is not None:
                z = torch.randn(real_data.size(0), 100).to(device)
                gen_data = gan(z)
                gen_target = torch.randint(0, 10, (real_data.size(0),)).to(device)
                data = torch.cat([real_data, gen_data])
                target = torch.cat([real_target, gen_target])
            else:
                data, target = real_data, real_target

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if ewc:
                loss += lambda_ewc * ewc.penalty(model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

def evaluate(model, dataloader, adversarial=False, attack=None, device='cpu'):
    model.eval()
    correct, total = 0, 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        if adversarial and attack is not None:
            data = attack.generate(data, target)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total * 100

# ========== 主程序 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
transform = transforms.Compose([transforms.ToTensor()])
task1_data = datasets.MNIST('.', train=True, download=True, transform=transform)
task2_data = datasets.MNIST('.', train=False, transform=transform)

task1_loader = DataLoader(task1_data, batch_size=64, shuffle=True)
task2_loader = DataLoader(task2_data, batch_size=64, shuffle=True)

# 初始化模型
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("=== Training Task1 ===")
train_task(model, optimizer, task1_loader, device=device)
acc_task1 = evaluate(model, task1_loader, device=device)
print(f"Task1 Initial Acc: {acc_task1:.2f}%")

# EWC
ewc = EWC(model, task1_loader, device=device)

# 训练 GAN
print("\n=== Training GAN on Task1 ===")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
train_gan(generator, discriminator, task1_loader, device=device)

# 任务2训练
print("\n=== Continual Learning Task2 ===")
train_task(model, optimizer, task2_loader, ewc=ewc, use_gan=True, gan=generator, device=device)
acc_task2 = evaluate(model, task2_loader, device=device)

# 对抗评估
attack = FGSM(model, epsilon=0.2)
adv_acc = evaluate(model, task1_loader, adversarial=True, attack=attack, device=device)

# 最终评估
print("\n==== Final Evaluation ====")
print(f"Task1 Forget Rate: {acc_task1 - evaluate(model, task1_loader, device=device):.2f}%")
print(f"Task2 Accuracy: {acc_task2:.2f}%")
print(f"Adversarial Accuracy on Task1: {adv_acc:.2f}%")
