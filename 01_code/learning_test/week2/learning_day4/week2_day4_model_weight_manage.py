# ==============================================
# Week2-Day4 水文反演模型权重管理【顶刊级优化+JSON修复版】
# 修复内容：JSON序列化问题，可完整归档元数据
# ==============================================
import os
import sys
import platform
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

# ==============================================
# 工具函数：处理numpy类型序列化问题
# ==============================================
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，解决JSON序列化问题"""
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

# ==============================================
# 1. 配置加载与环境初始化
# ==============================================
def load_config(config_path: str = "model_config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def init_environment(config: dict) -> tuple[dict, str]:
    base_dir = config["paths"]["base_dir"]
    dirs = [
        os.path.join(base_dir, config["paths"]["weight_save_subdir"]),
        os.path.join(base_dir, config["paths"]["log_subdir"]),
        os.path.join(base_dir, config["paths"]["plot_subdir"]),
        os.path.join(base_dir, config["paths"]["docs_subdir"])
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "yaml_version": yaml.__version__,
        "os": platform.system() + " " + platform.release(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return env_info, run_id

# ==============================================
# 2. 全链路随机种子固定
# ==============================================
def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# ==============================================
# 3. 分级日志系统
# ==============================================
def setup_logging(config: dict, run_id: str) -> logging.Logger:
    base_dir = config["paths"]["base_dir"]
    log_file = os.path.join(base_dir, config["paths"]["log_subdir"], f"model_training_{run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==============================================
# 4. 标准化数据集类
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
# 5. 可配置化水文反演模型
# ==============================================
class HydroInverseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "ReLU",
        batch_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation)()
        self.batch_norm = batch_norm

        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layer_blocks = []
            layer_blocks.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                layer_blocks.append(nn.BatchNorm1d(dim))
            layer_blocks.append(self.activation)
            setattr(self, f"layer{i+1}", nn.Sequential(*layer_blocks))
            prev_dim = dim
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.hidden_dims)):
            layer = getattr(self, f"layer{i+1}")
            x = layer(x)
        x = self.output_layer(x)
        return x

    def get_layer_names(self) -> list[str]:
        layer_names = [f"layer{i+1}" for i in range(len(self.hidden_dims))]
        layer_names.append("output_layer")
        return layer_names

# ==============================================
# 6. 水文反演标准评估指标计算
# ==============================================
def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

# ==============================================
# 7. 训练与验证核心函数
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
    metrics = calculate_metrics(all_preds, all_trues)
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
    metrics = calculate_metrics(all_preds, all_trues)
    return avg_loss, metrics, all_preds, all_trues

# ==============================================
# 8. 自动可视化
# ==============================================
def plot_train_curves(
    train_history: dict,
    val_history: dict,
    save_dir: str,
    run_id: str
):
    epochs = range(1, len(train_history["loss"]) + 1)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_history["loss"], label="Train MSE Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_history["loss"], label="Val MSE Loss", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Train & Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_history["r2"], label="Train R²", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_history["r2"], label="Val R²", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.title("Train & Validation R²", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"train_curves_{run_id}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_k_field_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    run_id: str,
    sample_num: int = 4
):
    grid_size = int(np.sqrt(y_true.shape[1]))
    sample_idx = np.random.choice(len(y_true), sample_num, replace=False)
    plt.figure(figsize=(16, 4*sample_num))
    for i, idx in enumerate(sample_idx):
        plt.subplot(sample_num, 2, 2*i+1)
        sns.heatmap(y_true[idx].reshape(grid_size, grid_size), cmap="viridis", vmin=1, vmax=5)
        plt.title(f"Sample {idx} - True Permeability Field", fontsize=12)
        plt.xlabel("Column", fontsize=10)
        plt.ylabel("Row", fontsize=10)
        plt.subplot(sample_num, 2, 2*i+2)
        sns.heatmap(y_pred[idx].reshape(grid_size, grid_size), cmap="viridis", vmin=1, vmax=5)
        plt.title(f"Sample {idx} - Inverted Permeability Field", fontsize=12)
        plt.xlabel("Column", fontsize=10)
        plt.ylabel("Row", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"k_field_comparison_{run_id}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ==============================================
# 9. Sim2Real层冻结与双维度严谨校验
# ==============================================
def freeze_layers_and_validate(
    model: HydroInverseNet,
    freeze_layer_names: list[str],
    logger: logging.Logger,
    device: torch.device
) -> HydroInverseNet:
    logger.info("="*80)
    logger.info("🔒 Sim2Real权重加载与层冻结验证（顶刊级严谨性校验）")
    logger.info(f"待冻结层：{freeze_layer_names}")
    logger.info("="*80)
    for layer_name in freeze_layer_names:
        if not hasattr(model, layer_name):
            logger.warning(f"层名{layer_name}不存在，跳过")
            continue
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = False
    logger.info("✅ 指定层冻结完成")
    logger.info("\n📌 各层参数requires_grad状态验证：")
    for layer_name in model.get_layer_names():
        layer = getattr(model, layer_name)
        grad_status = next(layer.parameters()).requires_grad
        logger.info(f"  {layer_name}: requires_grad={grad_status}")
    frozen_params_init = {}
    for layer_name in freeze_layer_names:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            frozen_params_init[layer_name] = [param.detach().cpu().clone() for param in layer.parameters()]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n📌 参数量统计：")
    logger.info(f"  总参数量：{total_params:,}")
    logger.info(f"  可训练参数量：{trainable_params:,}")
    logger.info(f"  冻结参数量：{total_params - trainable_params:,}")
    def validate_frozen_params_unchanged(model: HydroInverseNet):
        logger.info("\n🔍 冻结层参数数值最终校验：")
        all_unchanged = True
        for layer_name, init_params in frozen_params_init.items():
            layer = getattr(model, layer_name)
            current_params = [param.detach().cpu() for param in layer.parameters()]
            for init_p, current_p in zip(init_params, current_params):
                if not torch.allclose(init_p, current_p, atol=1e-8):
                    all_unchanged = False
                    logger.error(f"  {layer_name} 参数发生变化，冻结失效！")
                    break
        if all_unchanged:
            logger.info("✅ 所有冻结层参数完全未变化，冻结100%生效！")
        else:
            logger.error("❌ 冻结层参数发生变化，冻结失效！")
        return all_unchanged
    model.validate_frozen_params = validate_frozen_params_unchanged
    logger.info("\n✅ 层冻结与校验流程完成")
    logger.info("="*80)
    return model.to(device)

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    config = load_config("model_config.yaml")
    SEED = config["random_seed"]
    env_info, run_id = init_environment(config)
    set_random_seed(SEED)
    logger = setup_logging(config, run_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("="*80)
    logger.info("Week2-Day4 水文反演模型训练与权重管理【顶刊级优化+JSON修复版】")
    logger.info(f"运行ID：{run_id}")
    logger.info(f"随机种子：{SEED}，运行设备：{DEVICE}")
    logger.info(f"运行环境：{json.dumps(env_info, indent=2, ensure_ascii=False)}")
    logger.info("="*80)

    logger.info("🔍 正在加载数据集...")
    dataset = HydroInverseDataset(
        data_path=config["paths"]["dataset_path"],
        normalize=config["dataset"]["normalize"]
    )
    total_samples = len(dataset)
    logger.info(f"✅ 数据集加载完成，总样本量：{total_samples}")

    train_size = int(config["dataset"]["train_ratio"] * total_samples)
    val_size = int(config["dataset"]["val_ratio"] * total_samples)
    test_size = total_samples - train_size - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )
    logger.info(f"✅ 数据集划分完成：训练集{train_size}，验证集{val_size}，测试集{test_size}")

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

    if config["dataset"]["normalize"]:
        scaler_save_path = os.path.join(config["paths"]["base_dir"], config["paths"]["weight_save_subdir"], f"data_scalers_{run_id}.npy")
        dataset.save_scalers(scaler_save_path)
        logger.info(f"✅ 数据标准化器已保存至：{scaler_save_path}")

    model = HydroInverseNet(**config["model"]).to(DEVICE)
    logger.info("✅ 模型初始化完成")
    model_structure_path = os.path.join(config["paths"]["base_dir"], config["paths"]["docs_subdir"], f"model_architecture_{run_id}.txt")
    with open(model_structure_path, "w", encoding="utf-8") as f:
        f.write(str(model))
    logger.info(f"✅ 模型结构文档已保存至：{model_structure_path}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config["train"]["lr_scheduler_factor"],
        patience=config["train"]["lr_scheduler_patience"],
        verbose=True
    )
    early_stop_patience = config["train"]["early_stop_patience"]
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    train_history = {"loss": [], "rmse": [], "mae": [], "r2": []}
    val_history = {"loss": [], "rmse": [], "mae": [], "r2": []}
    best_weight_path = os.path.join(config["paths"]["base_dir"], config["paths"]["weight_save_subdir"], f"sim2real_pretrain_best_{run_id}.pth")

    logger.info(f"\n🚀 开始训练，总轮次：{config['train']['epochs']}")
    for epoch in tqdm(range(config["train"]["epochs"]), desc="训练进度"):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, DEVICE)
        for key in train_history.keys():
            train_history[key].append(train_metrics[key] if key != "loss" else train_loss)
            val_history[key].append(val_metrics[key] if key != "loss" else val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
                "random_seed": SEED
            }, best_weight_path)
            logger.info(f"Epoch {epoch+1}/{config['train']['epochs']} | 新最优权重已保存，验证损失：{best_val_loss:.6f}，验证R²：{val_metrics['r2']:.4f}")
        else:
            early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            logger.info(f"⏹️  早停触发，连续{early_stop_patience}轮验证损失未下降，停止训练")
            break
        if (epoch+1) % 5 == 0:
            logger.info(
                f"Epoch {epoch+1}/{config['train']['epochs']} | "
                f"训练损失：{train_loss:.6f} | 训练R²：{train_metrics['r2']:.4f} | "
                f"验证损失：{val_loss:.6f} | 验证R²：{val_metrics['r2']:.4f}"
            )

    logger.info(f"\n🎉 训练完成，最优轮次：{best_epoch+1}，最小验证损失：{best_val_loss:.6f}")
    logger.info(f"最优权重已保存至：{best_weight_path}")

    plot_dir = os.path.join(config["paths"]["base_dir"], config["paths"]["plot_subdir"])
    plot_train_curves(train_history, val_history, plot_dir, run_id)
    logger.info(f"✅ 训练曲线已保存至：{plot_dir}")

    logger.info("\n📊 开始测试集评估...")
    checkpoint = torch.load(best_weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, test_preds, test_trues = validate(model, test_loader, criterion, DEVICE)
    test_trues_phys = dataset.inverse_transform_k(test_trues)
    test_preds_phys = dataset.inverse_transform_k(test_preds)
    phys_valid = np.all((test_preds_phys >= 0.5) & (test_preds_phys <= 5.5))
    logger.info(f"✅ 物理合理性校验：{'通过，所有反演K值符合水文地质常识' if phys_valid else '未通过，存在异常K值'}")
    test_metrics_phys = calculate_metrics(test_preds_phys, test_trues_phys)
    logger.info(f"✅ 测试集评估完成（真实物理量纲）：")
    logger.info(f"  测试MSE：{test_metrics_phys['mse']:.6f}")
    logger.info(f"  测试RMSE：{test_metrics_phys['rmse']:.6f} m/d")
    logger.info(f"  测试MAE：{test_metrics_phys['mae']:.6f} m/d")
    logger.info(f"  测试R²：{test_metrics_phys['r2']:.4f}")

    plot_k_field_comparison(test_trues_phys, test_preds_phys, plot_dir, run_id)
    logger.info(f"✅ K场反演对比图已保存至：{plot_dir}")

    model = freeze_layers_and_validate(
        model=model,
        freeze_layer_names=config["finetune"]["freeze_layers"],
        logger=logger,
        device=DEVICE
    )

    logger.info("\n🔍 执行冻结层参数最终校验...")
    model.train()
    head_batch, k_batch = next(iter(train_loader))
    head_batch = head_batch.to(DEVICE)
    k_batch = k_batch.to(DEVICE)
    optimizer.zero_grad()
    outputs = model(head_batch)
    loss = criterion(outputs, k_batch)
    loss.backward()
    optimizer.step()
    model.validate_frozen_params(model)

    metadata = {
        "run_id": run_id,
        "config": config,
        "env_info": env_info,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics_physical": test_metrics_phys,
        "train_history": train_history,
        "val_history": val_history,
        "best_weight_path": best_weight_path,
        "model_structure_path": model_structure_path
    }
    metadata_serializable = convert_numpy_types(metadata)
    metadata_save_path = os.path.join(config["paths"]["base_dir"], config["paths"]["docs_subdir"], f"train_metadata_{run_id}.json")
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 全流程元数据已归档至：{metadata_save_path}")

    logger.info("\n" + "="*80)
    logger.info("🎉 Week2-Day4 模型权重管理全流程完成！")
    logger.info("✅ 已掌握：权重保存/加载、最优权重筛选、层冻结/解冻（Sim2Real核心）")
    logger.info("📊 所有训练指标、曲线、元数据已完整归档，可直接用于顶刊论文撰写")
    logger.info("📂 所有代码符合开源规范，可直接提交至GitHub仓库")
    logger.info("="*80)