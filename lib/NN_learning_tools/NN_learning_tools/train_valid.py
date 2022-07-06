import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# 1エポック学習するメソッド
# ----------------------------
# 入力: モデル，データローダー，損失関数，オプティマイザ，デバイス，現在のエポック数,
#       時間を測るかどうかのオプション
# 出力: エポックのロス，学習時間(要求があれば)
# 時間計測についてはもっとエレガントな書き方ないかな．．．
# ----------------------------
def train_epoch(model, trainloader, criterion, optimizer, device='cpu', 
        epoch=None, measure_time=False):
    train_loss = 0.0
    running_loss = 0.0

    # 時間計測をするならここで計測開始
    if measure_time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
    for count, item in enumerate(trainloader):
        inputs, labels = item
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        running_loss += loss.item()
        
        if count % 1000 == 0:
            print(f'#{epoch}, data:{count*4}, running_loss:{running_loss / 100:1.3f}')
            running_loss = 0.0
    
    # 時間計測終了
    if measure_time:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000
    else:
        elapsed_time = None
    
    train_loss /= len(trainloader)
    return train_loss, elapsed_time

# テストデータで検証するメソッド
# ----------------------------
# 入力: モデル，データローダー, 損失関数，デバイス，時間計測をするかどうかのオプション
# 出力: ロスと精度，計測時間(要求があれば)
# ----------------------------
def valid(model, testloader, criterion, device='cpu', measure_time=False):
    with torch.no_grads():
        total = 0
        correct = 0
        test_loss = 0

        # 時間計測をするならここで計測開始
        if measure_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # ロスの計算
            test_loss += criterion(outputs, labels).item()
            # 正答率の計算
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()
        
        # 時間計測終了
        if measure_time:
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000
        else:
            elapsed_time = None
        
        test_loss /= len(testloader)
        test_accuracy = correct / total

    return test_loss, test_accuracy, elapsed_time
        

