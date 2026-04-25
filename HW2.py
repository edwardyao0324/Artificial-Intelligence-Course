from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import shutil

# 把檔案放過去給 Evaluator
src = r"C:\Users\Edward\Desktop\Artificial Intelligence Course\bloodmnist.npz"
dst_dir = r"C:\Users\Edward\.medmnist"
dst = os.path.join(dst_dir, 'bloodmnist.npz')
os.makedirs(dst_dir, exist_ok=True)
if not os.path.exists(dst):
    shutil.copy(src, dst)
    print("已複製 bloodmnist.npz 到 .medmnist 資料夾")


plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False
plt.ion()  # 不用案關閉就可以跳出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")


NUM_EPOCHS = 30
BATCH_SIZE = 128
ROOT = r"C:\Users\Edward\Desktop\Artificial Intelligence Course"
OUTPUT = r"C:\Users\Edward\Desktop\Artificial Intelligence Course\output3"
os.makedirs(OUTPUT, exist_ok=True)

data_flag = 'bloodmnist'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

# 仔入資料
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(split='train', transform=train_transform, download=False, root=ROOT)
val_dataset   = DataClass(split='val',   transform=val_test_transform, download=False, root=ROOT)
test_dataset  = DataClass(split='test',  transform=val_test_transform, download=False, root=ROOT)

train_loader         = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,   shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
val_loader           = data.DataLoader(dataset=val_dataset,   batch_size=2*BATCH_SIZE, shuffle=False)
test_loader          = data.DataLoader(dataset=test_dataset,  batch_size=2*BATCH_SIZE, shuffle=False)


# EDA 視覺化
fig1 = train_dataset.montage(length=1)
fig1.save(os.path.join(OUTPUT, 'montage_1.png'))
fig2 = train_dataset.montage(length=20)
fig2.save(os.path.join(OUTPUT, 'montage_20.png'))

#定義CNN
class Net(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.0):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 優化器
def get_criterion():
    if task == "multi-label, binary-class":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()

def get_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSProp':
        return optim.RMSprop(model.parameters(), lr=lr)

def get_val_accuracy_from(model):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)
            y_true = torch.cat((y_true, targets.cpu()), 0)
            y_score = torch.cat((y_score, outputs.cpu()), 0)
    y_score = y_score.detach().numpy()
    evaluator = Evaluator(data_flag, 'val')
    metrics = evaluator.evaluate(y_score)
    return metrics[1]

def evaluate(model, split):
    model.eval()
    y_true  = torch.tensor([])
    y_score = torch.tensor([])
    y_pred_list = []
    y_true_list = []

    if split == 'train':
        loader = train_loader_at_eval
    elif split == 'val':
        loader = val_loader
    else:
        loader = test_loader

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                preds   = outputs.argmax(dim=-1)
                y_pred_list.extend(preds.cpu().numpy())
                y_true_list.extend(targets.cpu().numpy())
                targets = targets.float().resize_(len(targets), 1)

            y_true  = torch.cat((y_true,  targets.cpu()), 0)
            y_score = torch.cat((y_score, outputs.cpu()), 0)

    y_true  = y_true.numpy()
    y_score = y_score.detach().numpy()

    evaluator = Evaluator(data_flag, split)
    metrics   = evaluator.evaluate(y_score)
    acc = metrics[1]
    auc = metrics[0]
    f1  = f1_score(y_true_list, y_pred_list, average='macro')

    print(f'{split}  auc: {auc:.3f}  acc: {acc:.3f}  macro-F1: {f1:.3f}')

    # 混淆矩陣
    cm   = confusion_matrix(y_true_list, y_pred_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'混淆矩陣（{split}）')
    plt.savefig(os.path.join(OUTPUT, f'confusion_matrix_{split}.png'))
    plt.draw()
    plt.pause(0.1)

    return acc, f1

def train_and_eval(lr, optimizer_name='SGD', num_epochs=10, dropout_rate=0.0, label=''):
    model     = Net(in_channels=n_channels, num_classes=n_classes, dropout_rate=dropout_rate).to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, optimizer_name, lr)

    best_val_acc    = 0
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f'[{label}] epoch {epoch+1}/{num_epochs}'):
            inputs  = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        val_acc = get_val_accuracy_from(model)
        val_acc_history.append(val_acc)
        print(f'  [{label}] epoch {epoch+1}  val_acc: {val_acc:.3f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT, f'best_{label}.pt'))

    return val_acc_history, model

def write_log(round_id, prompt, proposed_config, adopted, adopted_reason, val_accuracy, macro_f1, reflection):
    with open(os.path.join(OUTPUT, 'agent_log.md'), 'a', encoding='utf-8') as f:
        f.write(f"""
## Round {round_id}
**prompt:** "{prompt}"

**proposed_config:**
{proposed_config}

**adopted:** {adopted}
**adopted_reason:** "{adopted_reason}"

**val_accuracy:** {val_accuracy:.3f}
**macro_f1:** {macro_f1:.3f}

**reflection:** "{reflection}"

---
""")

#Baseline CNN
print("\n" + "="*50)
print("PART A：Baseline CNN")
print("="*50)

best_val_acc   = 0
baseline_model = Net(in_channels=n_channels, num_classes=n_classes).to(device)
criterion      = get_criterion()
optimizer      = get_optimizer(baseline_model, 'SGD', lr=0.001)
baseline_val_history = []

for epoch in range(NUM_EPOCHS):
    baseline_model.train()
    for inputs, targets in tqdm(train_loader, desc=f'Baseline epoch {epoch+1}/{NUM_EPOCHS}'):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = baseline_model(inputs)
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    val_acc = get_val_accuracy_from(baseline_model)
    baseline_val_history.append(val_acc)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}  val_acc: {val_acc:.3f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(baseline_model.state_dict(), os.path.join(OUTPUT, 'best_model.pt'))
        print(f'  → 儲存最佳模型 (val_acc={val_acc:.3f})')

print('\n==> Baseline 評估 ...')
evaluate(baseline_model, 'train')
val_acc, val_f1 = evaluate(baseline_model, 'val')
evaluate(baseline_model, 'test')

write_log(
    round_id=1,
    prompt="你是我的 CNN experiment agent。這是 baseline 結果。請提出下一輪最值得做的設定變更，只能從 lr, optimizer, batch_size,dropout, channels, wd, augmentation中選 1-2 項。",
    proposed_config="- optimizer: SGD\n- lr: 0.001\n- dropout: 0.0",
    adopted="true",
    adopted_reason="第一輪先建立 baseline，不做任何改動",
    val_accuracy=val_acc,
    macro_f1=val_f1,
    reflection="第一輪 baseline，觀察 confusion matrix 決定下一步"
)

#Learning Rate 比較
print("\n" + "="*50)
print("PART B 實驗2：Learning Rate 比較")
print("="*50)

lr_list    = [0.01, 0.001, 0.0001]
lr_results = {}

for lr_val in lr_list:
    label = f'SGD_lr{lr_val}'
    history, _ = train_and_eval(lr=lr_val, optimizer_name='SGD', num_epochs=10, label=label)
    lr_results[label] = history

plt.figure(figsize=(8, 5))
for label, history in lr_results.items():
    plt.plot(history, label=label)
plt.xlabel('訓練輪數')
plt.ylabel('驗證準確率')
plt.savefig(os.path.join(OUTPUT, 'lr_comparison.png'))
plt.title('Learning Rate 比較（SGD）')
plt.legend()
plt.tight_layout()
plt.draw()
plt.pause(0.1)

#Optimizer 比較
print("\n" + "="*50)
print("PART B 實驗3：Optimizer 比較")
print("="*50)

opt_list    = ['SGD', 'Adam', 'RMSProp']
opt_results = {}

for opt_name in opt_list:
    label = f'{opt_name}_lr0.001'
    history, _ = train_and_eval(lr=0.001, optimizer_name=opt_name, num_epochs=10, label=label)
    opt_results[label] = history

plt.figure(figsize=(8, 5))
for label, history in opt_results.items():
    plt.plot(history, label=label)
plt.xlabel('訓練輪數')
plt.savefig(os.path.join(OUTPUT, 'optimizer_comparison.png'))
plt.ylabel('驗證準確率')
plt.title('Optimizer 比較（lr=0.001）')
plt.legend()
plt.tight_layout()
plt.draw()
plt.pause(0.1)

#Ablation（Dropout）
print("\n" + "="*50)
print("PART B 實驗4：Ablation（Dropout）")
print("="*50)

dropout_list    = [0.0, 0.3, 0.5]
dropout_results = {}

for dr in dropout_list:
    label = f'dropout{dr}'
    history, _ = train_and_eval(lr=0.001, optimizer_name='SGD', num_epochs=10, dropout_rate=dr, label=label)
    dropout_results[label] = history

plt.figure(figsize=(8, 5))
for label, history in dropout_results.items():
    plt.plot(history, label=label)
plt.savefig(os.path.join(OUTPUT, 'dropout_ablation.png'))
plt.xlabel('訓練輪數')
plt.ylabel('驗證準確率')
plt.title('Dropout Ablation（SGD lr=0.001）')
plt.legend()
plt.tight_layout()
plt.draw()
plt.pause(0.1)

# Part B
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for label, history in lr_results.items():
    axes[0].plot(history, label=label)
axes[0].set_title('Learning Rate 比較')
axes[0].set_xlabel('訓練輪數')
axes[0].set_ylabel('驗證準確率')
axes[0].legend()

for label, history in opt_results.items():
    axes[1].plot(history, label=label)
axes[1].set_title('Optimizer 比較')
axes[1].set_xlabel('訓練輪數')
axes[1].set_ylabel('驗證準確率')
axes[1].legend()

for label, history in dropout_results.items():
    axes[2].plot(history, label=label)
axes[2].set_title('Dropout Ablation')
plt.savefig(os.path.join(OUTPUT, 'part_b_overview.png'))
axes[2].set_xlabel('訓練輪數')
axes[2].set_ylabel('驗證準確率')
axes[2].legend()

plt.suptitle('Part B 實驗比較總覽', fontsize=14)
plt.tight_layout()
plt.draw()
plt.pause(0.1)

print(f"\n輸出檔案在：{OUTPUT}")
print("==> 全部完成！")

plt.ioff()  
plt.show()