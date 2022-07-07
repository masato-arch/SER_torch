import matplotlib.pyplot as plt

# 一定エポックごとにログを取るメソッド
def plot_log(train_losses, test_losses, train_accuracy, test_accuracy, epoch=None):
    print(f'plot_log: #{epoch}, train_accuracy:{train_accuracy[-1]}, test_accuracy:{test_accuracy[-1]}')
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend()
    
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='train_accuracy')
    plt.plot(test_accuracy, label='test_accuracy')
    plt.legend()
    
    plt.show()