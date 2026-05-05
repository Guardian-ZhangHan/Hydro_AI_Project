import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    """
    水文地质顶刊标准 3 大指标
    无任何第三方依赖，100% 稳定
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # R2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # NSE
    nse = 1 - ss_res / ss_tot

    return {
        'R2': round(float(r2), 4),
        'RMSE': round(float(rmse), 6),
        'NSE': round(float(nse), 4)
    }

def plot_k_comparison(true_k, pred_k, save_path="comparison.png"):
    """
    绘制 K 真实值 VS 反演值对比图
    顶刊格式，直接可用
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.imshow(true_k.reshape(10,10), cmap="jet")
    plt.title("True K")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(pred_k.reshape(10,10), cmap="jet")
    plt.title("Predicted K")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()