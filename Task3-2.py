import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

x1 = np.array([166, 176, 171, 173, 169], dtype=float)
y_true1 = np.array([58, 75, 62, 70, 60], dtype=float)

x = (x1 - np.mean(x1)) / np.std(x1)
y_true = (y_true1 - np.mean(y_true1)) / np.std(y_true1)

n = 5
learning_rate = 0.001

def train(method):
    w = 0
    b = 0
    state = {} # 儲存優化器狀態
    loss_history = []
    param_history = []
    arrow = [400, 800, 1200, 1600, 2000]

    for i in range(2000):
        y_pred = w * x + b
        loss = np.mean((y_pred - y_true) ** 2)

        dw = (2 / n) * np.sum((y_pred - y_true) * x)
        db = (2 / n) * np.sum(y_pred - y_true)

        if method == "sgd":
            w -= learning_rate * dw
            b -= learning_rate * db

        elif method == "momentum":
            beta = 0.9
            v_w = state.get("v_w", 0)
            v_b = state.get("v_b", 0)
            v_w = beta * v_w + (1 - beta) * dw
            v_b = beta * v_b + (1 - beta) * db
            w -= learning_rate * v_w
            b -= learning_rate * v_b
            state["v_w"] = v_w
            state["v_b"] = v_b

        elif method == "adagrad":
            G_w = state.get("G_w", 0)
            G_b = state.get("G_b", 0)
            G_w += dw ** 2
            G_b += db ** 2
            w -= learning_rate * dw / (np.sqrt(G_w) + 1e-8)
            b -= learning_rate * db / (np.sqrt(G_b) + 1e-8)
            state["G_w"] = G_w
            state["G_b"] = G_b

        elif method == "rmsprop":
            beta = 0.9
            G_w = state.get("G_w", 0)
            G_b = state.get("G_b", 0)
            G_w = beta * G_w + (1 - beta) * dw ** 2
            G_b = beta * G_b + (1 - beta) * db ** 2
            w -= learning_rate * dw / (np.sqrt(G_w) + 1e-8)
            b -= learning_rate * db / (np.sqrt(G_b) + 1e-8)
            state["G_w"] = G_w
            state["G_b"] = G_b

        elif method == "adam":
            beta1 = 0.9
            beta2 = 0.999
            m_w = state.get("m_w", 0)
            v_w = state.get("v_w", 0)
            m_b = state.get("m_b", 0)
            v_b = state.get("v_b", 0)
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * dw ** 2
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * db ** 2
            w -= learning_rate * m_w / (np.sqrt(v_w) + 1e-8)
            b -= learning_rate * m_b / (np.sqrt(v_b) + 1e-8)
            state["m_w"] = m_w
            state["v_w"] = v_w
            state["m_b"] = m_b
            state["v_b"] = v_b

        loss_history.append(loss)
        if (i + 1) in arrow:
            param_history.append((i + 1, w, b))

        if (i+1) % 400 == 0:
            print(f"迭代={i+1}, w={w}, b={b}, loss={loss}") #i+1因為i從0開始，印出來會少1

    return w, b, loss_history, param_history


methods = ["sgd", "momentum", "adagrad", "rmsprop", "adam"]

# ★ 一個視窗，5列 × 2欄
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 20))
fig.suptitle("各優化器訓練結果", fontsize=16, fontweight='bold')

for row, m in enumerate(methods):
    print(f"現在訓練：{m}")
    w, b, loss_history, param_history = train(m)

    # 左欄：線性回歸結果
    ax1 = axes[row, 0]
    ax1.scatter(x, y_true, color='green', label="數據")
    ax1.plot(x, w * x + b, color='red', label="擬合直線")
    ax1.set_title(f"{m} 線性回歸結果")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    # 右欄：Loss 收斂曲線
    ax2 = axes[row, 1]
    ax2.plot(loss_history, label="Loss")
    ax2.set_title(f"{m} Loss 收斂曲線")
    ax2.set_xlabel("迭代")
    ax2.set_ylabel("損失")

    for epoch, w_i, b_i in param_history:
        ax2.annotate(
            f"{epoch}\nw={w_i:.4f}\nb={b_i:.4f}",
            xy=(epoch - 1, loss_history[epoch - 1]),
            xytext=(epoch - 1, loss_history[epoch - 1] + 0.05),
            arrowprops=dict(arrowstyle='->', lw=0.5),
            fontsize=8,
            ha='center'
        )

    ax2.legend()

plt.tight_layout()
plt.show()