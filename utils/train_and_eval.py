import torch.optim.lr_scheduler
from tqdm import tqdm
from torch.nn import functional as F


def dice_score(preds, targets):
    preds = F.sigmoid(preds)
    preds = (preds > 0.5).float()
    score = (2. * (preds * targets).sum()) / (preds + targets).sum()
    return torch.mean(score).item()

def create_lr_scheduler(optimizer, num_step, num_epochs, warmup: bool=True, warmup_epochs=1, warmup_factor=1e3):
    assert num_step > 0 and num_epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            # 因为当已经越过warmup的epoch时，希望学习率逐步从1 下降到 0
            return (1 - (x - warmup_epochs * num_step) / ((num_epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train_one_epoch(model, optimizer, criterion,data_loader, device, lr_scheduler, scaler=None):
    # 训练一个epoch
    model.train()
    train_total_loss = 0
    train_iterations = 0

    for idx, data in enumerate(tqdm(data_loader)):
        train_iterations += 1
        train_img = data[0].to(device)
        train_mask = data[1].to(device)

        optimizer.zero_grad()

        train_output_mask = model(train_img)
        train_loss = criterion(train_output_mask['out'], train_mask)
        train_total_loss += train_loss.item()

        train_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if idx == 2:
            break

    train_epoch_loss = train_total_loss / train_iterations
    return train_epoch_loss

def evaluate(model, criterion, data_loader, device):
    model.eval()
    with torch.no_grad():
        valid_total_loss = 0
        valid_iterations = 0
        scores = 0

        for vidx, val_data in enumerate(tqdm(data_loader)):
            valid_iterations += 1
            val_img, val_mask = val_data[0].to(device), val_data[1].to(device)
            pred = model(val_img)
            val_loss = criterion(pred['out'], val_mask)
            valid_total_loss += val_loss.item()
            scores += dice_score(pred['out'], val_mask)
            if vidx == 2:
                break

    val_epoch_loss = valid_total_loss / valid_iterations
    dice_epoch_score = scores / valid_iterations
    return val_epoch_loss, dice_epoch_score