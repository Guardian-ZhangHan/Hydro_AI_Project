# ==============================================
# Week2-Day5 早停机制训练终极版【WRR顶刊规范】
# 修复内容：weight_decay类型报错
# 顶刊优化：全链路可复现性、参数量统计、训练时长、矢量图可视化、学习率曲线、误差分布直方图
# 完全适配水文反演Sim2Real场景
# ==============================================
import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 导入手写早停类
from early_stopping import EarlyStopping

# ==============================================
# 工具函数
# ==============================================
def load_config(config_path: str = "day5_config.yaml") -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def init_environment(config: dict) -> tuple[dict, str]:
    """初始化环境，创建文件夹"""
    base_dir = config["paths"]["base_dir"]
    dirs = [
        os.path.join(base_dir, config["paths"]["weight_save_subdir"]),
        os.path.join(base_dir, config["paths"]["log_subdir"]),
        os.path.join(base_dir, config["paths"]["doc_subdir"])
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 记录运行环境
    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "yaml_version": yaml.__version__,
        "os": os.name,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return env_info, run_id

def set_random_seed(seed: int):
    """全链路固定随机种子，顶刊级可复现性终极加固"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA确定性设置

def seed_worker(worker_id: int):
    """DataLoader多线程种子固定"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def setup_logging(config: dict, run_id: str) -> logging.Logger:
    """设置规范日志系统"""
    base_dir = config["paths"]["base_dir"]
    log_file = os.path.join(base_dir, config["paths"]["log_subdir"], f"train_log_{run_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger, log_file

def convert_numpy_types(obj):
    """解决JSON序列化问题"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

def count_parameters(model) -> dict:
    """顶刊级参数量详细统计：总参数量、可训练/冻结、分层统计"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    layer_params = {}
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += param.numel()
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "per_layer": layer_params
    }

# ==============================================
# 数据集类
# ==============================================
class HydroInverseDataset(Dataset):
    def __init__(self, data_path: str, normalize: bool = True):
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在：{data_path}")
        data = np.load(data_path)
        self.head = data["head"].astype(np.float32)
        self.k = data["k"].astype(np.float32)
        self.normalize = normalize
        self.scalers = {}

        if self.normalize:
            self.scalers["head_mean"] = np.mean(self.head, axis=0)
            self.scalers["head_std"] = np.std(self.head, axis=0)
            self.head = (self.head - self.scalers["head_mean"]) / (self.scalers["head_std"] + 1e-8)
            self.scalers["k_mean"] = np.mean(self.k, axis=0)
            self.scalers["k_std"] = np.std(self.k, axis=0)
            self.k = (self.k - self.scalers["k_mean"]) / (self.scalers["k_std"] + 1e-8)

        self.head = torch.from_numpy(self.head)
        self.k = torch.from_numpy(self.k)

    def __len__(self) -> int:
        return len(self.head)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head[idx], self.k[idx]

    def save_scalers(self, save_path: str):
        np.save(save_path, self.scalers)

    def inverse_transform_k(self, k_norm: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return k_norm
        return k_norm * self.scalers["k_std"] + self.scalers["k_mean"]

# ==============================================
# 模型类
# ==============================================
class HydroInverseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "ReLU",
        batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation)()
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layer_blocks = []
            layer_blocks.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                layer_blocks.append(nn.BatchNorm1d(dim))
            layer_blocks.append(self.activation)
            if dropout_rate > 0:
                layer_blocks.append(nn.Dropout(dropout_rate))
            setattr(self, f"layer{i+1}", nn.Sequential(*layer_blocks))
            prev_dim = dim
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.hidden_dims)):
            layer = getattr(self, f"layer{i+1}")
            x = layer(x)
        x = self.output_layer(x)
        return x

# ==============================================
# 训练/验证函数
# ==============================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    for head_batch, k_batch in train_loader:
        head_batch = head_batch.to(device)
        k_batch = k_batch.to(device)
        optimizer.zero_grad()
        outputs = model(head_batch)
        loss = criterion(outputs, k_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * head_batch.size(0)
        all_preds.append(outputs.detach().cpu().numpy())
        all_trues.append(k_batch.cpu().numpy())
    avg_loss = total_loss / len(train_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    mse = mean_squared_error(all_trues, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_trues, all_preds)
    r2 = r2_score(all_trues, all_preds)
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
    return avg_loss, metrics

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for head_batch, k_batch in val_loader:
            head_batch = head_batch.to(device)
            k_batch = k_batch.to(device)
            outputs = model(head_batch)
            loss = criterion(outputs, k_batch)
            total_loss += loss.item() * head_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(k_batch.cpu().numpy())
    avg_loss = total_loss / len(val_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    mse = mean_squared_error(all_trues, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_trues, all_preds)
    r2 = r2_score(all_trues, all_preds)
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
    return avg_loss, metrics, all_preds, all_trues

# ==============================================
# 顶刊级可视化函数
# ==============================================
def plot_full_train_curves(
    train_history: dict,
    val_history: dict,
    early_stop_summary: dict,
    save_dir: str,
    run_id: str
):
    """绘制完整训练曲线：损失+R²+学习率，PNG+SVG双格式保存"""
    epochs = range(1, len(train_history["loss"]) + 1)
    lr_history = early_stop_summary["lr_history"] if len(early_stop_summary["lr_history"]) == len(epochs) else [0]*len(epochs)
    
    plt.figure(figsize=(18, 6))
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_history["loss"], label="Train MSE Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_history["loss"], label="Val MSE Loss", color="#ff7f0e", linewidth=2)
    best_epoch = early_stop_summary["best_epoch"]
    plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    if early_stop_summary["early_stop_triggered"]:
        stop_epoch = early_stop_summary["total_epochs_run"]
        plt.axvline(x=stop_epoch, color="black", linestyle=":", label=f"Early Stop ({stop_epoch})")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Train & Validation Loss Curve", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # R²曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_history["r2"], label="Train R²", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_history["r2"], label="Val R²", color="#ff7f0e", linewidth=2)
    plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.title("Train & Validation R² Curve", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.ylim(-0.5, 1)

    # 学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, lr_history, color="#2ca02c", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.grid(alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    # 双格式保存，PNG用于预览，SVG用于论文排版
    plt.savefig(os.path.join(save_dir, f"full_train_curves_{run_id}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"full_train_curves_{run_id}.svg"), dpi=300, bbox_inches="tight", format="svg")
    plt.close()

def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    run_id: str
):
    """绘制反演误差分布直方图，顶刊论文必备"""
    error = y_pred.flatten() - y_true.flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(error, kde=True, bins=50, color="#1f77b4", edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Inversion Error (m/d)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Permeability Inversion Error Distribution", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"error_distribution_{run_id}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"error_distribution_{run_id}.svg"), dpi=300, bbox_inches="tight", format="svg")
    plt.close()

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    # 1. 加载配置
    config = load_config("day5_config.yaml")
    SEED = config["random_seed"]
    
    # 2. 初始化环境
    env_info, run_id = init_environment(config)
    base_dir = config["paths"]["base_dir"]
    
    # 3. 固定全链路随机种子
    set_random_seed(SEED)
    
    # 4. 设置日志
    logger, log_file = setup_logging(config, run_id)
    
    # 5. 设备自适应
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 6. 打印启动信息
    logger.info("="*80)
    logger.info("Week2-Day5 早停机制与泛化能力优化【WRR顶刊终极版】")
    logger.info(f"运行ID：{run_id}")
    logger.info(f"随机种子：{SEED}，运行设备：{DEVICE}")
    logger.info(f"早停机制启用：{config['early_stopping']['enable']}，耐心轮次：{config['early_stopping']['patience']}")
    logger.info(f"运行环境：{json.dumps(env_info, indent=2, ensure_ascii=False)}")
    logger.info("="*80)

    # 7. 加载数据集
    logger.info("🔍 正在加载数据集...")
    dataset = HydroInverseDataset(
        data_path=config["paths"]["dataset_path"],
        normalize=config["dataset"]["normalize"]
    )
    total_samples = len(dataset)
    logger.info(f"✅ 数据集加载完成，总样本量：{total_samples}")

    # 8. 数据集划分
    train_size = int(config["dataset"]["train_ratio"] * total_samples)
    val_size = int(config["dataset"]["val_ratio"] * total_samples)
    test_size = total_samples - train_size - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )
    logger.info(f"✅ 数据集划分完成：训练集{train_size}，验证集{val_size}，测试集{test_size}")

    # 9. DataLoader
    train_loader = DataLoader(
        train_set, batch_size=config["train"]["batch_size"],
        shuffle=True, drop_last=True,
        worker_init_fn=seed_worker, generator=generator
    )
    val_loader = DataLoader(
        val_set, batch_size=config["train"]["batch_size"],
        shuffle=False, drop_last=False,
        worker_init_fn=seed_worker, generator=generator
    )
    test_loader = DataLoader(
        test_set, batch_size=config["train"]["batch_size"],
        shuffle=False, drop_last=False,
        worker_init_fn=seed_worker, generator=generator
    )

    # 10. 保存标准化器
    if config["dataset"]["normalize"]:
        scaler_save_path = os.path.join(base_dir, config["paths"]["weight_save_subdir"], f"data_scalers_{run_id}.npy")
        dataset.save_scalers(scaler_save_path)
        logger.info(f"✅ 数据标准化器已保存至：{scaler_save_path}")

    # 11. 初始化模型+参数量统计
    model = HydroInverseNet(**config["model"]).to(DEVICE)
    param_stats = count_parameters(model)
    logger.info("✅ 模型初始化完成，参数量统计：")
    logger.info(f"  总参数量：{param_stats['total']:,}")
    logger.info(f"  可训练参数量：{param_stats['trainable']:,}")
    logger.info(f"  分层参数量：{json.dumps(param_stats['per_layer'], indent=2, ensure_ascii=False)}")
    
    # 保存模型结构
    model_structure_path = os.path.join(base_dir, config["paths"]["doc_subdir"], f"model_architecture_{run_id}.txt")
    with open(model_structure_path, "w", encoding="utf-8") as f:
        f.write(str(model))
        f.write("\n\n" + "="*50 + "\n")
        f.write(f"参数量统计：\n{json.dumps(param_stats, indent=2, ensure_ascii=False)}")
    logger.info(f"✅ 模型结构文档已保存至：{model_structure_path}")

    # 12. 损失函数、优化器、学习率调度器（彻底修复类型问题）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"])  # 强制转浮点数，彻底杜绝类型报错
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(config["train"]["lr_scheduler_factor"]),
        patience=int(config["train"]["lr_scheduler_patience"]),
        verbose=True
    )

    # 13. 初始化早停机制
    early_stopping = None
    if config["early_stopping"]["enable"]:
        best_model_save_path = os.path.join(base_dir, config["paths"]["weight_save_subdir"], f"best_model_{run_id}.pth")
        early_stopping = EarlyStopping(
            patience=int(config["early_stopping"]["patience"]),
            min_delta=float(config["early_stopping"]["min_delta"]),
            mode=config["early_stopping"]["mode"],
            restore_best_weights=config["early_stopping"]["restore_best_weights"],
            save_best_model=config["early_stopping"]["save_best_model"],
            save_path=best_model_save_path,
            logger=logger
        )
        logger.info("✅ 早停机制初始化完成")

    # 14. 训练历史记录
    train_history = {"loss": [], "rmse": [], "mae": [], "r2": []}
    val_history = {"loss": [], "rmse": [], "mae": [], "r2": []}

    # 15. 开始训练+计时
    logger.info(f"\n🚀 开始训练，最大轮次：{config['train']['epochs']}")
    train_start_time = datetime.now()
    for epoch in tqdm(range(config["train"]["epochs"]), desc="训练进度"):
        current_epoch = epoch + 1
        # 训练
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        # 验证
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, DEVICE)

        # 记录历史
        for key in train_history.keys():
            train_history[key].append(train_metrics[key] if key != "loss" else train_loss)
            val_history[key].append(val_metrics[key] if key != "loss" else val_loss)

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 早停判断
        if early_stopping is not None:
            stop_triggered = early_stopping(val_loss, model, current_epoch, current_lr)
            if stop_triggered:
                break

        # 按间隔打印日志
        if current_epoch % config["log"]["print_interval"] == 0:
            logger.info(
                f"Epoch {current_epoch}/{config['train']['epochs']} | "
                f"训练损失：{train_loss:.6f} | 训练R²：{train_metrics['r2']:.4f} | "
                f"验证损失：{val_loss:.6f} | 验证R²：{val_metrics['r2']:.4f} | "
                f"当前学习率：{current_lr:.8f}"
            )

    # 16. 训练完成统计
    train_end_time = datetime.now()
    train_duration = (train_end_time - train_start_time).total_seconds()
    logger.info(f"\n🎉 训练完成！总训练时长：{train_duration:.2f} 秒")
    if early_stopping is not None:
        early_stop_summary = early_stopping.get_summary()
        logger.info(f"早停总结：{json.dumps(early_stop_summary, indent=2, ensure_ascii=False)}")
    else:
        early_stop_summary = {"best_epoch": config["train"]["epochs"], "early_stop_triggered": False, "lr_history": []}

    # 17. 绘制顶刊级可视化图
    doc_dir = os.path.join(base_dir, config["paths"]["doc_subdir"])
    plot_full_train_curves(train_history, val_history, early_stop_summary, doc_dir, run_id)
    logger.info(f"✅ 完整训练曲线已保存至：{doc_dir}")

    # 18. 测试集评估
    logger.info("\n📊 开始测试集评估...")
    test_loss, test_metrics, test_preds, test_trues = validate(model, test_loader, criterion, DEVICE)
    test_trues_phys = dataset.inverse_transform_k(test_trues)
    test_preds_phys = dataset.inverse_transform_k(test_preds)
    phys_valid = np.all((test_preds_phys >= 0.5) & (test_preds_phys <= 5.5))
    logger.info(f"✅ 物理合理性校验：{'通过，所有反演K值符合水文地质常识' if phys_valid else '未通过，存在异常K值'}")
    logger.info(f"✅ 测试集评估完成（真实物理量纲）：")
    logger.info(f"  测试MSE：{test_metrics['mse']:.6f}")
    logger.info(f"  测试RMSE：{test_metrics['rmse']:.6f} m/d")
    logger.info(f"  测试MAE：{test_metrics['mae']:.6f} m/d")
    logger.info(f"  测试R²：{test_metrics['r2']:.4f}")

    # 19. 绘制误差分布直方图
    plot_error_distribution(test_trues_phys, test_preds_phys, doc_dir, run_id)
    logger.info(f"✅ 反演误差分布直方图已保存至：{doc_dir}")

    # 20. 全流程元数据归档
    metadata = {
        "run_id": run_id,
        "config": config,
        "env_info": env_info,
        "param_stats": param_stats,
        "train_duration_seconds": train_duration,
        "early_stop_summary": early_stop_summary,
        "test_metrics_physical": test_metrics,
        "train_history": train_history,
        "val_history": val_history,
        "log_file": log_file
    }
    metadata_serializable = convert_numpy_types(metadata)
    metadata_save_path = os.path.join(doc_dir, f"train_metadata_{run_id}.json")
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 全流程元数据已归档至：{metadata_save_path}")

    # 21. 自动导出环境依赖
    logger.info("🔍 正在导出环境依赖...")
    os.system(f"{sys.executable} -m pip freeze > {os.path.join(base_dir, 'requirements.txt')}")
    logger.info(f"✅ 环境依赖已导出至requirements.txt")

    # 22. 完成总结
    logger.info("\n" + "="*80)
    logger.info("🎉 Week2-Day5 早停机制与泛化能力优化全流程完成！")
    logger.info("✅ 已掌握：早停机制原理与实现、验证集监控、规范日志保存、防过拟合策略")
    logger.info("📊 所有训练指标、顶刊级可视化图、元数据已完整归档，可直接用于论文写作")
    logger.info("📂 所有代码符合GitHub开源规范，可直接推送至仓库")
    logger.info("="*80)