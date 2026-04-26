import os

# ====================== 这里固定你的项目绝对根目录（D盘你的项目，一字不改！） ======================
PROJECT_ROOT = r"D:\Hydro_Ai_Project"

# 原图要求的全部规范目录清单，完全不变
dir_list = [
    "01_data/01_raw_study_area",
    "01_data/02_synthetic_dataset",
    "02_code/01_modflow_model",
    "02_code/02_sim2real_model",
    "03_models/pretrained_weight",
    "03_models/scaler",
    "04_results/figures_vector",
    "04_results/figures_raster",
    "05_literature"
]

# 批量创建，强制全部生成在 D:\Hydro_Ai_Project 里面
for relative_path in dir_list:
    # 拼接完整绝对路径，强制绑定你的D盘项目
    full_dir = os.path.join(PROJECT_ROOT, relative_path)
    os.makedirs(full_dir, exist_ok=True)
    print(f"【强制创建到D盘项目】文件夹 {relative_path} 完整路径：{full_dir}  创建完成")