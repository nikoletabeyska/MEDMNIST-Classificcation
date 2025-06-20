import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator, BloodMNIST
from models import ResNet18, ResNet50
import wandb
from tqdm import trange

def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, as_rgb, model_path, run):
    lr = 0.001
    gamma = 0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]
    size = 128  # For bloodmnist_128
    patience = 10  # Number of epochs to wait for improvement before stopping

    if data_flag != 'bloodmnist':
        raise ValueError("This script is configured for bloodmnist only")

    # Get dataset info from MedMNIST
    info = INFO[data_flag]
    n_channels = info['n_channels']  # 3 for BloodMNIST (RGB)
    n_classes = len(info['label'])   # 8 classes for BloodMNIST
    task = info['task']              # multi-class

    # GPU setup
    str_ids = gpu_ids.split(',')
    gpu_ids = [int(id) for id in str_ids if int(id) >= 0]
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    os.makedirs(output_root, exist_ok=True)

    print('==> Preparing data...')

    # Data transforms for 128x128 images
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels)
    ])

    # Load BloodMNIST dataset from MedMNIST
    train_dataset = BloodMNIST(split='train', transform=data_transform, download=download, size=128, as_rgb=as_rgb)
    val_dataset = BloodMNIST(split='val', transform=data_transform, download=download, size=128, as_rgb=as_rgb)
    test_dataset = BloodMNIST(split='test', transform=data_transform, download=download, size=128, as_rgb=as_rgb)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print('==> Building and training model...')

    # Initialize model
    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    # Evaluators
    train_evaluator = Evaluator(data_flag, 'train', size=size)
    val_evaluator = Evaluator(data_flag, 'val', size=size)
    test_evaluator = Evaluator(data_flag, 'test', size=size)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize W&B
    wandb.init(
        project="bloodmnist_128-experiment",
        config={
            "learning_rate": lr,
            "gamma": gamma,
            "milestones": milestones,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "model_flag": model_flag,
            "dataset": data_flag,
            "image_size": size,
            "n_channels": n_channels,
            "n_classes": n_classes,
            "run_name": run,
            "patience": patience
        }
    )
    wandb.run.name = f"{data_flag}_{model_flag}_{run}"

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)
        print(f'train auc: {train_metrics[1]:.5f} acc: {train_metrics[2]:.5f}')
        print(f'val auc: {val_metrics[1]:.5f} acc: {val_metrics[2]:.5f}')
        print(f'test auc: {test_metrics[1]:.5f} acc: {test_metrics[2]:.5f}')
        wandb.log({
            "pretrained_train_loss": train_metrics[0],
            "pretrained_train_auc": train_metrics[1],
            "pretrained_train_acc": train_metrics[2],
            "pretrained_val_loss": val_metrics[0],
            "pretrained_val_auc": val_metrics[1],
            "pretrained_val_acc": val_metrics[2],
            "pretrained_test_loss": test_metrics[0],
            "pretrained_test_auc": test_metrics[1],
            "pretrained_test_acc": test_metrics[2]
        })

    if num_epochs == 0:
        wandb.finish()
        return

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_' + log for log in logs]
    val_logs = ['val_' + log for log in logs]
    test_logs = ['test_' + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)
    trigger_times = 0  # Counter for early stopping

    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, task, criterion, optimizer, device, epoch)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)

        scheduler.step()

        # Monitor GPU utilization
        gpu_memory = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
        gpu_util = torch.cuda.utilization(device) if hasattr(torch.cuda, 'utilization') else -1

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        # Add GPU metrics to log_dict
        log_dict['gpu_memory_allocated_gb'] = gpu_memory
        log_dict['gpu_memory_reserved_gb'] = gpu_memory_reserved
        if gpu_util != -1:
            log_dict['gpu_utilization_percent'] = gpu_util

        wandb.log(log_dict, step=epoch)

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            trigger_times = 0  # Reset trigger times on improvement
            print(f'cur_best_auc: {best_auc}')
            print(f'cur_best_epoch: {best_epoch}')
            model_path = os.path.join(output_root, 'best_model.pth')
            torch.save({'net': best_model.state_dict()}, model_path)
            artifact = wandb.Artifact(f'best_model_{run}_epoch_{best_epoch}', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
        else:
            trigger_times += 1
            print(f'Early stopping trigger: {trigger_times}/{patience}')
            if trigger_times >= patience:
                print(f'Early stopping triggered at epoch {epoch} with best AUC: {best_auc}')
                break

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run, output_root)

    wandb.log({
        "final_train_loss": train_metrics[0],
        "final_train_auc": train_metrics[1],
        "final_train_acc": train_metrics[2],
        "final_val_loss": val_metrics[0],
        "final_val_auc": val_metrics[1],
        "final_val_acc": val_metrics[2],
        "final_test_loss": test_metrics[0],
        "final_test_auc": test_metrics[1],
        "final_test_acc": test_metrics[2],
        "best_epoch": best_epoch,
        "best_val_auc": best_auc
    })

    log = f'{data_flag}\n' + \
          f'train auc: {train_metrics[1]:.5f} acc: {train_metrics[2]:.5f}\n' + \
          f'val auc: {val_metrics[1]:.5f} acc: {val_metrics[2]:.5f}\n' + \
          f'test auc: {test_metrics[1]:.5f} acc: {test_metrics[2]:.5f}\n'
    print(log)

    with open(os.path.join(output_root, f'{data_flag}_log.txt'), 'a') as f:
        f.write(log)

    wandb.finish()

def train(model, train_loader, task, criterion, optimizer, device, epoch):
    total_loss = []
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss = sum(total_loss) / len(total_loss)
    wandb.log({'train_loss_logs': epoch_loss}, step=epoch)
    return epoch_loss

def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):
    model.eval()
    total_loss = []
    y_score = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs)
            total_loss.append(loss.item())
            y_score.append(outputs.cpu())

        y_score = torch.cat(y_score, dim=0).numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        test_loss = sum(total_loss) / len(total_loss)
        return [test_loss, auc, acc]

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='RUN Baseline model for BloodMNIST 128x128')
    parser.add_argument('--data_flag',
                        default='bloodmnist',
                        type=str,
                        help='Dataset flag (must be bloodmnist)')
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root for models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='number of training epochs',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=64,
                        help='batch size for training',
                        type=int)
    parser.add_argument('--download',
                        action='store_true',
                        help='download dataset if not present')
    parser.add_argument('--as_rgb',
                        action='store_true',
                        help='use RGB images (default for bloodmnist)')
    parser.add_argument('--model_path',
                        default=None,
                        help='path to pretrained model',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone: resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='name for evaluation run',
                        type=str)

    args = parser.parse_args()
    if args.data_flag != 'bloodmnist':
        raise ValueError("This script is configured for bloodmnist only")
    main(args.data_flag, args.output_root, args.num_epochs, args.gpu_ids, args.batch_size,
         args.download, args.model_flag, args.as_rgb, args.model_path, args.run)
