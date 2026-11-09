import numpy as np

np.random.seed(0)

# ---------- 1. 造一个最土 MLP ----------
n_in, n_hid, n_out = 4, 6, 1
W1 = np.random.randn(n_in, n_hid) * 0.5   # 输入→隐藏 (shadow weight, FP32)
W2 = np.random.randn(n_hid, n_out) * 0.5  # 隐藏→输出 (shadow weight, FP32)
lr = 0.01

# ---------- 2. 三值化函数 ----------
def ternarize(W):
    """absmean 缩放 + round 截断到 {-1,0,+1}"""
    s = np.mean(np.abs(W), axis=0, keepdims=True)   # 每列一个缩放因子
    s = np.clip(s, 1e-5, None)                       # 防除零
    Q = np.round(W / s).clip(-1, 1)                  # 三值
    return Q, s

# ---------- 3. STE 直通估计器 ----------
def ste_forward(W):
    """前向用三值，反向把梯度原样透传给浮点 W"""
    Q, s = ternarize(W)
    # trick: 用 (W - W.detach) + Q 的思路，NumPy 里直接返回 Q 即可，
    # 因为后面反向我们手动把梯度赋给 W，不经过 Q 求导。
    return Q, s

# ---------- 4. 一次训练迭代 ----------
def train_step(x, y, W1, W2, lr):
    # --- 前向 ---
    q1, s1 = ste_forward(W1)          # 隐藏层权重临时三值化
    h_pre = x @ q1                     # 矩阵乘 → 实际是加/减/跳过
    h = np.maximum(h_pre, 0)           # ReLU（和普通 MLP 一样）

    q2, s2 = ste_forward(W2)          # 输出层权重临时三值化
    y_pred = h @ q2                    # 再一轮加减法
    loss = 0.5 * (y_pred - y) ** 2

    # --- 反向（STE：梯度直接透传给浮点权重）---
    dy = y_pred - y                    # dL/dy_pred
    dW2 = h.T @ dy                     # 假装输出层是普通线性层
    dh = dy @ q2.T                     # 传回隐藏层
    dh_pre = dh * (h_pre > 0)          # ReLU 梯度
    dW1 = x.T @ dh_pre                 # 假装隐藏层是普通线性层

    # --- 更新的是浮点 shadow weight ---
    W1 -= lr * dW1
    W2 -= lr * dW2
    return loss, W1, W2

# ---------- 5. 跑 5 步看效果 ----------
x = np.array([[1.0, -0.5, 2.0, 0.3]])   # 一个样本
y = np.array([[3.0]])                    # 目标

for i in range(5):
    loss, W1, W2 = train_step(x, y, W1, W2, lr)
    print(f"step {i}: loss = {loss[0,0]:.4f}")

# ---------- 6. 看看权重真变成三值没 ----------
Q1, s1 = ternarize(W1)
print("\n隐藏层三值权重 Q1:\n", Q1)
print("缩放因子 s1:", s1)
print("唯一出现的值:", np.unique(Q1))   # 应该只有 -1, 0, 1

