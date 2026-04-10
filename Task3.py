import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei' #中文
plt.rcParams['axes.unicode_minus'] = False  #負號

x1 = np.array([166,176,171,173,169], dtype=float)
y_true1 = np.array([58,75,62,70,60], dtype=float)

x = (x1 - np.mean(x1)) / np.std(x1) #標準化，變-1~1的分佈 
y_true = (y_true1 - np.mean(y_true1)) / np.std(y_true1) #標準化，變-1~1的分佈 

w=0
b=0
n=5
learning_rate=0.001
loss_history = []
param_history = []  # 記錄特定的 w, b, loss
arrow = [400, 800, 1200, 1600, 2000] # 在這些迭代上加箭頭和註解

for i in range(2000):
    linear1 = w * x + b #線性回歸不斷塞入
    loss =np.mean((linear1 - y_true)**2) #不斷塞入新數值
    
    dw=(2/n)*np.sum((w*x+b-y_true)*x)
    db=(2/n)*np.sum(w*x+b-y_true) 
    
    w= w - learning_rate*dw       
    b= b - learning_rate*db
    
    loss_history.append(loss)
    if (i+1) in arrow:
        param_history.append((i+1, w, b))
    if (i+1) % 400 == 0:
        print(f"迭代={i+1}, w={w}, b={b}, loss={loss}") #i+1因為i從0開始，印出來會少1


plt.figure()
plt.scatter(x, y_true, color='green', label="數據") # 散佈註解
plt.plot(x, w*x + b, color='red', label="擬合直線")
plt.title("線性回歸結果")
plt.xlabel("x")
plt.ylabel("y")


plt.figure()
plt.title("Loss 隨 epoch 下降的收斂曲線") 
plt.xlabel("迭代")
plt.ylabel("損失")
plt.plot(loss_history, color='blue', label="Loss")

#以下大多AI打的
for epoch, w, b, in param_history:
    plt.annotate(
        f"迭代={epoch}\nw={w:.4f}\nb={b:.4f}",
        xy=(epoch - 1, loss_history[epoch - 1]),
        xytext=(epoch - 1, loss_history[epoch - 1] + 0.05),
        arrowprops=dict(arrowstyle='->', lw=0.5),
        fontsize=8,
        ha='center'
    )
plt.legend()
plt.show()

