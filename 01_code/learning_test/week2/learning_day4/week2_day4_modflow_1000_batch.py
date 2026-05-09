# ==============================================
# Week2-Day4 1000组MODFLOW批量建模【顶刊级完整修复版】
# 修复内容：
# 1. 放宽水力梯度校验阈值，避免均匀场模型误判
# 2. 优化校验逻辑，数值稳定性更强
# 包含所有优化点：
# - YAML配置文件 + 命令行参数化
# - 模块化函数设计
# - 物理合理性数据校验（含水力梯度）
# - 运行环境全记录
# - 数据集SHA256哈希校验
# - 断点续跑功能
# - 中间文件自动清理
# - 数据集统计信息自动生成
# - 数据集可视化
# - 双格式元数据保存
# ==============================================
import flopy
import numpy as np
import os
import sys
import platform
import json
import yaml
import hashlib
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI问题
import matplotlib.pyplot as plt

# ==============================================
# 【优化1】命令行参数解析
# ==============================================
def parse_args():
    parser = argparse.ArgumentParser(description="MODFLOW batch modeling for sim2real dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--k_min", type=float, default=None, help="Override minimum hydraulic conductivity (m/d)")
    parser.add_argument("--k_max", type=float, default=None, help="Override maximum hydraulic conductivity (m/d)")
    parser.add_argument("--n_models", type=int, default=None, help="Override number of models to run")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()
    return args

# ==============================================
# 【优化2】YAML配置文件加载
# ==============================================
def load_config(config_path, args):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置文件
    if args.k_min is not None:
        config["model"]["k_min"] = args.k_min
    if args.k_max is not None:
        config["model"]["k_max"] = args.k_max
    if args.n_models is not None:
        config["model"]["total_models"] = args.n_models
    if args.seed is not None:
        config["model"]["random_seed"] = args.seed
    
    return config

# ==============================================
# 【优化3】环境初始化与运行环境全记录
# ==============================================
def init_environment(config):
    """初始化环境，创建文件夹，记录运行环境信息"""
    base_dir = config["paths"]["base_dir"]
    dirs = [
        os.path.join(base_dir, config["paths"]["model_output_subdir"]),
        os.path.join(base_dir, config["paths"]["dataset_save_subdir"]),
        os.path.join(base_dir, config["paths"]["docs_subdir"]),
        os.path.join(base_dir, config["paths"]["log_subdir"]),
        os.path.join(base_dir, config["paths"]["checkpoint_subdir"]),
        os.path.join(base_dir, config["paths"]["plot_subdir"])
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 记录运行环境信息（顶刊可复现性强制要求）
    env_info = {
        "python_version": sys.version,
        "flopy_version": flopy.__version__,
        "numpy_version": np.__version__,
        "yaml_version": yaml.__version__,
        "os": platform.system() + " " + platform.release(),
        "mf6_exe_path": config["paths"]["mf6_exe_path"],
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return env_info

# ==============================================
# 【优化4】日志系统升级
# ==============================================
def setup_logging(config):
    """设置日志系统"""
    base_dir = config["paths"]["base_dir"]
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(base_dir, config["paths"]["log_subdir"], f"modflow_batch_{current_time}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_file

# ==============================================
# 【优化5】物理合理性数据校验（已修复阈值）
# ==============================================
def validate_data(well_head, k_flat, head_data, config):
    """
    校验数据的物理合理性：
    1. 水头值应在左右定水头之间（90-100m）
    2. K值应在设定范围内
    3. 无NaN/Inf值
    4. 水力梯度不超过阈值（已放宽）
    """
    if not config["optimization"]["data_validation"]:
        return True
    
    # 校验1：无NaN/Inf
    if np.isnan(well_head).any() or np.isinf(well_head).any():
        return False
    if np.isnan(k_flat).any() or np.isinf(k_flat).any():
        return False
    
    # 校验2：水头在合理范围内（边界是100和90，允许微小波动）
    if (well_head < 88.0).any() or (well_head > 102.0).any():
        return False
    
    # 校验3：K值在设定范围内
    k_min = config["model"]["k_min"]
    k_max = config["model"]["k_max"]
    if (k_flat < k_min * 0.8).any() or (k_flat > k_max * 1.2).any():
        return False
    
    # 校验4：水力梯度不超过阈值（已放宽到0.1，避免均匀场误判）
    max_grad = config["optimization"]["max_hydraulic_gradient"]
    delr = config["model"]["delr"]
    delc = config["model"]["delc"]
    
    grad_x = np.gradient(head_data, axis=2) / delr
    grad_y = np.gradient(head_data, axis=1) / delc
    max_calc_grad = np.max(np.sqrt(grad_x**2 + grad_y**2))
    
    if max_calc_grad > max_grad:
        return False
    
    return True

# ==============================================
# 【优化6】MODFLOW中间文件自动清理
# ==============================================
def clean_up_temp_files(model_ws, model_name, config):
    """清理MODFLOW生成的中间文件，仅保留水头文件(.hds)"""
    if not config["optimization"]["clean_up_temp_files"]:
        return
    
    # 保留的文件扩展名
    keep_extensions = [".hds", ".nam", ".mfsim"]
    
    for filename in os.listdir(model_ws):
        file_path = os.path.join(model_ws, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in keep_extensions:
                try:
                    os.remove(file_path)
                except:
                    pass  # 忽略删除错误

# ==============================================
# 【优化7】断点续跑功能
# ==============================================
def load_checkpoint(config):
    """加载检查点，返回已运行的模型ID和数据"""
    base_dir = config["paths"]["base_dir"]
    checkpoint_path = os.path.join(base_dir, config["paths"]["checkpoint_subdir"], "checkpoint.npz")
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        return (
            data["last_model_idx"].item(),
            list(data["head_list"]),
            list(data["k_list"]),
            list(data["failed_ids"])
        )
    else:
        return -1, [], [], []

def save_checkpoint(config, last_model_idx, head_list, k_list, failed_ids):
    """保存检查点"""
    base_dir = config["paths"]["base_dir"]
    checkpoint_path = os.path.join(base_dir, config["paths"]["checkpoint_subdir"], "checkpoint.npz")
    np.savez_compressed(
        checkpoint_path,
        last_model_idx=last_model_idx,
        head_list=head_list,
        k_list=k_list,
        failed_ids=failed_ids
    )

# ==============================================
# 【优化8】数据集SHA256哈希校验
# ==============================================
def get_file_hash(file_path):
    """计算文件的SHA256哈希值"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# ==============================================
# 【优化9】数据集统计信息自动生成
# ==============================================
def generate_dataset_stats(head_np, k_np, config):
    """生成数据集的统计信息"""
    stats = {
        "head": {
            "mean": float(np.mean(head_np)),
            "std": float(np.std(head_np)),
            "min": float(np.min(head_np)),
            "max": float(np.max(head_np))
        },
        "k": {
            "mean": float(np.mean(k_np)),
            "std": float(np.std(k_np)),
            "min": float(np.min(k_np)),
            "max": float(np.max(k_np))
        }
    }
    return stats

# ==============================================
# 【优化10】数据集可视化
# ==============================================
def plot_sample(head_data, k_data, idx, save_dir, config):
    """绘制单个样本的水头场和K场"""
    nrow = config["model"]["nrow"]
    ncol = config["model"]["ncol"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制K场
    im1 = ax1.imshow(k_data.reshape(nrow, ncol), cmap="viridis")
    plt.colorbar(im1, ax=ax1, label="Hydraulic Conductivity (m/d)")
    ax1.set_title(f"Permeability Field (Model {idx})")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    
    # 绘制水头场
    im2 = ax2.imshow(head_data.reshape(nrow, ncol), cmap="Blues")
    plt.colorbar(im2, ax=ax2, label="Head (m)")
    ax2.set_title(f"Head Field (Model {idx})")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{idx:04d}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ==============================================
# 核心模型构建函数
# ==============================================
def build_and_run_model(model_id: int, k_field: np.ndarray, config, logger):
    base_dir = config["paths"]["base_dir"]
    model_name = f"sim_model_{model_id:04d}"
    model_ws = os.path.join(base_dir, config["paths"]["model_output_subdir"], model_name)
    os.makedirs(model_ws, exist_ok=True)

    for retry in range(config["model"]["max_retry"] + 1):
        try:
            # 构建模型
            sim = flopy.mf6.MFSimulation(
                sim_name=model_name, sim_ws=model_ws, 
                exe_name=config["paths"]["mf6_exe_path"], version="mf6"
            )
            tdis = flopy.mf6.ModflowTdis(
                sim, time_units="DAYS", nper=1, 
                perioddata=[(config["model"]["perlen"], config["model"]["nstp"], config["model"]["tsmult"])]
            )
            # 适配MODFLOW 6.7.0的print_option
            ims = flopy.mf6.ModflowIms(
                sim, print_option="SUMMARY",
                outer_dvclose=1e-6, outer_maximum=1000,
                inner_maximum=100, inner_dvclose=1e-6
            )
            gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, save_flows=True)
            dis = flopy.mf6.ModflowGwfdis(
                gwf, nlay=config["model"]["nlay"], nrow=config["model"]["nrow"], ncol=config["model"]["ncol"],
                delr=config["model"]["delr"], delc=config["model"]["delc"], 
                top=config["model"]["top"], botm=config["model"]["botm"]
            )
            sto = flopy.mf6.ModflowGwfsto(
                gwf, ss=1e-5, sy=0.1, steady_state=config["model"]["steady"]
            )
            npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k_field)
            ic = flopy.mf6.ModflowGwfic(gwf, strt=95.0)
            
            # 定水头边界
            chd_spd = []
            for row in range(config["model"]["nrow"]):
                chd_spd.append([(0, row, 0), 100.0])
                chd_spd.append([(0, row, config["model"]["ncol"]-1), 90.0])
            chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: chd_spd})
            oc = flopy.mf6.ModflowGwfoc(
                gwf, head_filerecord=f"{model_name}.hds", 
                saverecord=[("HEAD", "LAST")]
            )

            # 写入并运行模型
            sim.write_simulation()
            success, buff = sim.run_simulation(silent=False, report=False)

            if success:
                # 读取水头结果
                head_file = flopy.utils.HeadFile(os.path.join(model_ws, f"{model_name}.hds"))
                head_data = head_file.get_data()
                well_head = np.array(
                    [head_data[tuple(well)] for well in config["model"]["monitor_wells"]], 
                    dtype=np.float32
                )
                k_flat = k_field.flatten().astype(np.float32)
                
                # 物理合理性数据校验
                if validate_data(well_head, k_flat, head_data, config):
                    # 清理中间文件
                    clean_up_temp_files(model_ws, model_name, config)
                    return True, well_head, k_flat, head_data
                else:
                    logger.warning(f"模型{model_id}数据未通过物理合理性校验，丢弃")
                    return False, None, None, None
            else:
                if retry < config["model"]["max_retry"]:
                    logger.warning(f"模型{model_id}第{retry+1}次运行失败，重试中...")
                    continue
                else:
                    logger.error(f"模型{model_id}运行失败")
                    return False, None, None, None
        except Exception as e:
            logger.error(f"模型{model_id}运行异常：{str(e)}")
            return False, None, None, None

# ==============================================
# 【优化11】顶刊级元数据文档生成
# ==============================================
def generate_metadata_doc(config, env_info, dataset_stats, success_count, dataset_hash, log_file):
    """生成顶刊级元数据文档"""
    base_dir = config["paths"]["base_dir"]
    doc_path = os.path.join(base_dir, config["paths"]["docs_subdir"], "MODFLOW仿真数据集元数据文档.md")
    
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(f"""# MODFLOW 6地下水仿真数据集元数据文档
## 1. 基本信息
| 项目 | 内容 |
|------|------|
| 数据集名称 | sim2real_full_dataset.npz |
| 生成时间 | {env_info['run_time']} |
| 总模型数 | {config['model']['total_models']} |
| 有效样本数 | {success_count} |
| 随机种子 | {config['model']['random_seed']} |
| 数据集SHA256哈希 | {dataset_hash} |

## 2. 运行环境（可复现性）
| 项目 | 内容 |
|------|------|
| Python版本 | {env_info['python_version']} |
| FloPy版本 | {env_info['flopy_version']} |
| NumPy版本 | {env_info['numpy_version']} |
| PyYAML版本 | {env_info['yaml_version']} |
| 操作系统 | {env_info['os']} |
| MODFLOW 6路径 | {env_info['mf6_exe_path']} |
| 运行日志 | {log_file} |

## 3. 模型设置
- 网格尺寸：{config['model']['nrow']}行×{config['model']['ncol']}列×{config['model']['nlay']}层
- 网格步长：{config['model']['delr']}m×{config['model']['delc']}m
- 含水层高程范围：{config['model']['botm']}~{config['model']['top']}m
- 边界条件：左右两侧定水头（左100m，右90m）
- 模拟类型：稳态模拟
- 模拟时长：{config['model']['perlen']}天
- 渗透系数范围：{config['model']['k_min']}~{config['model']['k_max']} m/d
- 储水系数Ss：1e-5 1/m
- 给水度Sy：0.1
- 监测井数量：{len(config['model']['monitor_wells'])}口，位置：{config['model']['monitor_wells']}
- 最大允许水力梯度：{config['optimization']['max_hydraulic_gradient']} m/m

## 4. 数据集统计信息
### 4.1 监测井水头数据（输入）
| 统计量 | 数值 |
|--------|------|
| 均值 | {dataset_stats['head']['mean']:.4f} m |
| 标准差 | {dataset_stats['head']['std']:.4f} m |
| 最小值 | {dataset_stats['head']['min']:.4f} m |
| 最大值 | {dataset_stats['head']['max']:.4f} m |

### 4.2 渗透系数场数据（标签）
| 统计量 | 数值 |
|--------|------|
| 均值 | {dataset_stats['k']['mean']:.4f} m/d |
| 标准差 | {dataset_stats['k']['std']:.4f} m/d |
| 最小值 | {dataset_stats['k']['min']:.4f} m/d |
| 最大值 | {dataset_stats['k']['max']:.4f} m/d |

## 5. 数据集结构
| 数组名 | 形状 | 含义 |
|--------|------|------|
| head | ({success_count}, {len(config['model']['monitor_wells'])}) | 模型输入：{len(config['model']['monitor_wells'])}口监测井的稳态水头值 |
| k | ({success_count}, {config['model']['nrow']*config['model']['ncol']}) | 模型标签：{config['model']['nrow']}×{config['model']['ncol']}网格的渗透系数场（展平） |
| monitor_wells | ({len(config['model']['monitor_wells'])}, 3) | 监测井的空间坐标 |
| grid_size | (2,) | 模型网格尺寸 |
| random_seed | 标量 | 随机种子，保证实验100%可复现 |
| dataset_stats | dict | 数据集统计信息 |

## 6. 可复现性说明
1.  所有模型的边界条件、网格设置、监测井位置完全固定，仅渗透系数场为唯一变量，严格符合单变量控制实验规范
2.  每个模型使用独立的随机种子（主种子+模型ID），所有随机生成的渗透系数场可完全复现
3.  所有模型运行日志、代码、元数据均已完整归档，符合FAIR科学数据原则
4.  数据经过物理合理性校验（水头范围、K值范围、水力梯度），无异常值
5.  代码与数据集已同步至GitHub仓库：https://github.com/Guardian-ZhangHan/Hydro_AI_Project
""")
    return doc_path

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 加载配置文件
    config = load_config(args.config, args)
    
    # 3. 初始化环境
    env_info = init_environment(config)
    
    # 4. 设置日志
    logger, log_file = setup_logging(config)
    
    # 5. 打印启动信息
    logger.info("="*80)
    logger.info("Week2-Day4 1000组MODFLOW批量建模【顶刊级完整修复版】")
    logger.info(f"配置文件：{args.config}")
    logger.info(f"项目路径：{config['paths']['base_dir']}")
    logger.info(f"随机种子：{config['model']['random_seed']}，总模型数：{config['model']['total_models']}")
    logger.info(f"运行环境：{json.dumps(env_info, indent=2, ensure_ascii=False)}")
    logger.info("="*80)
    
    # 6. 启动前校验
    if not os.path.exists(config["paths"]["mf6_exe_path"]):
        logger.critical(f"❌ mf6.exe不存在！路径：{config['paths']['mf6_exe_path']}")
        exit(1)
    else:
        logger.info(f"✅ mf6.exe路径验证通过：{config['paths']['mf6_exe_path']}")
    
    # 7. 测试模型
    logger.info("🔍 先运行测试模型，验证环境...")
    test_k = np.ones((config["model"]["nlay"], config["model"]["nrow"], config["model"]["ncol"])) * 3.0
    test_success, _, _, _ = build_and_run_model(9999, test_k, config, logger)
    if not test_success:
        logger.critical("❌ 测试模型运行失败！")
        exit(1)
    else:
        logger.info("✅ 测试模型运行成功！开始批量建模...")
    
    # 8. 加载检查点
    last_model_idx, head_list, k_list, failed_ids = load_checkpoint(config)
    start_idx = last_model_idx + 1
    if start_idx > 0:
        logger.info(f"🔄 从检查点恢复，从模型{start_idx}继续运行...")
    
    # 9. 批量运行
    head_data_list = []  # 用于可视化
    for model_idx in tqdm(range(start_idx, config["model"]["total_models"]), desc="批量建模进度"):
        # 生成随机K场（每个模型独立随机种子）
        np.random.seed(config["model"]["random_seed"] + model_idx)
        k_random = np.random.uniform(
            low=config["model"]["k_min"], high=config["model"]["k_max"], 
            size=(config["model"]["nlay"], config["model"]["nrow"], config["model"]["ncol"])
        )
        
        # 运行模型
        success, well_head, k_flat, head_data = build_and_run_model(model_idx, k_random, config, logger)
        
        if success:
            head_list.append(well_head)
            k_list.append(k_flat)
            # 保存用于可视化的样本
            if model_idx in config["optimization"]["plot_samples"]:
                head_data_list.append((model_idx, head_data, k_flat))
        else:
            failed_ids.append(model_idx)
        
        # 定期保存检查点
        if (model_idx + 1) % config["optimization"]["checkpoint_interval"] == 0:
            save_checkpoint(config, model_idx, head_list, k_list, failed_ids)
            logger.info(f"💾 检查点已保存（模型{model_idx+1}）")
    
    # 10. 结果统计
    success_count = len(head_list)
    head_np = np.array(head_list, dtype=np.float32)
    k_np = np.array(k_list, dtype=np.float32)
    
    logger.info(f"\n✅ 批量建模完成，成功运行{success_count}/{config['model']['total_models']}个模型")
    if len(failed_ids) > 0:
        logger.warning(f"失败模型ID：{failed_ids}")
    
    # 11. 保存数据集
    if success_count > 0:
        # 生成数据集统计信息
        dataset_stats = generate_dataset_stats(head_np, k_np, config)
        logger.info(f"📊 数据集统计信息：{json.dumps(dataset_stats, indent=2, ensure_ascii=False)}")
        
        # 保存数据集
        base_dir = config["paths"]["base_dir"]
        dataset_path = os.path.join(base_dir, config["paths"]["dataset_save_subdir"], "sim2real_full_dataset.npz")
        np.savez_compressed(
            dataset_path,
            head=head_np,
            k=k_np,
            monitor_wells=config["model"]["monitor_wells"],
            grid_size=(config["model"]["nrow"], config["model"]["ncol"]),
            random_seed=config["model"]["random_seed"],
            total_models=config["model"]["total_models"],
            success_models=success_count,
            dataset_stats=dataset_stats
        )
        logger.info(f"✅ 数据集已保存至：{dataset_path}")
        logger.info(f"数据集维度：head={head_np.shape}, k={k_np.shape}")
        
        # 计算数据集SHA256哈希
        dataset_hash = get_file_hash(dataset_path)
        logger.info(f"🔒 数据集SHA256哈希：{dataset_hash}")
        
        # 保存元数据（双格式）
        metadata = {
            "config": config,
            "env_info": env_info,
            "dataset_stats": dataset_stats,
            "total_models": config["model"]["total_models"],
            "success_models": success_count,
            "failed_model_ids": failed_ids,
            "dataset_hash": dataset_hash,
            "log_file": log_file
        }
        metadata_path = os.path.join(base_dir, config["paths"]["docs_subdir"], "dataset_metadata.npy")
        np.save(metadata_path, metadata)
        # 同时保存JSON格式，方便查看
        metadata_json_path = os.path.join(base_dir, config["paths"]["docs_subdir"], "dataset_metadata.json")
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 元数据已保存至：{metadata_path}（JSON格式：{metadata_json_path}）")
        
        # 生成顶刊级元数据文档
        doc_path = generate_metadata_doc(config, env_info, dataset_stats, success_count, dataset_hash, log_file)
        logger.info(f"✅ 顶刊级元数据文档已生成：{doc_path}")
        
        # 数据集可视化
        plot_dir = os.path.join(base_dir, config["paths"]["plot_subdir"])
        logger.info(f"🎨 正在生成数据集可视化样本...")
        for idx, head_data, k_data in head_data_list:
            plot_sample(head_data, k_data, idx, plot_dir, config)
        logger.info(f"✅ 数据集可视化样本已保存至：{plot_dir}")
        
        # 清理检查点
        checkpoint_path = os.path.join(base_dir, config["paths"]["checkpoint_subdir"], "checkpoint.npz")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("✅ 检查点已清理")
    else:
        logger.critical("❌ 没有成功运行的模型！")

    logger.info("\n" + "="*80)
    logger.info("🎉 批量建模全流程完成！")
    logger.info("="*80)