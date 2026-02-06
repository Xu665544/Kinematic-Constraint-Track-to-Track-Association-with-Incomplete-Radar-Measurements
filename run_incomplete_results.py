import os
from pathlib import Path
import numpy as np
import pandas as pd

from ttta_adsb_radar_kcdtw import CostParams, run_ttta_for_scene

# ============================
# 路径与场景配置
# ============================

# 自动拼接 "/SceneX"
BASE_DATA_DIR_ROOT = r"C:\Users\Administrator\Desktop\小论文\加噪数据"   
RESULT_DIR         = r"C:\Users\Administrator\Desktop\小论文\results_Incomplete_Measurment"

# 运行 Scene 1 到 9
# SCENES = list(range(1, 10))
SCENES = [7]
# 运行所有噪声等级 0 到 10
LEVELS = list(range(0, 11))          

# 定义需要处理的缺失类型（不包含 'none'）
# MISS_TYPES = ['theta', 'phi', 'speed', 'course']
MISS_TYPES = ['r']
# 所有场景的雷达位置字典r
RADAR_LLA_ALL = {
    1: (31.2, 121.6, 20.0),
    2: (4.0,  101.0, 20.0),
    3: (23, 112, 20.0),
    4: (25.45, 55.56, 20.0),
    5: (40.5, 28.0,  20.0),
    6: (35.0, 138.5, 20.0),
    7: (27.0, 120.5, 20.0),
    8: (51.2, 5.2,   20.0),
    9: (34.0, 126.6, 20.0),       #注意 !!!!!!!!!!!!!!
}

# ============================
# 算法参数配置
# ============================

COMMON_ARGS = dict(
    sigma_r=200.0, sigma_theta_deg=1.0, sigma_phi_deg=1.0,
    sigma_v=20.0, sigma_psi_deg=5.0,
    sigma_dr=100.0, sigma_dtheta=2.0, sigma_dphi=2.0,
    sigma_dv=50.0, sigma_dpsi=2.0,
    kc_weight=1.0
)

# Baseline: time_gate_sec = 60
PARAMS_BASE = CostParams(
    **COMMON_ARGS,
    time_gate_sec=60
)

# KC-IMDTW: time_gate_sec = 100
PARAMS_KC = CostParams(
    **COMMON_ARGS,
    time_gate_sec=100
)

# ============================

def _pairs_to_df(pairs):
    rows = []
    for p in pairs:
        # 兼容 (radar_id, adsb_id) 或 (radar_id, adsb_id, cost)
        if len(p) == 3:
            r, a, c = p
        else:
            r, a = p
            c = np.nan
        rows.append({
            "radar_id": str(r),
            "adsb_id": str(a),
            "cost": float(c) if c is not None else np.nan,
            "correct": 1 if str(r) == str(a) else 0
        })
    return pd.DataFrame(rows)

def _safe_acc(df):
    if df is None or len(df) == 0:
        return np.nan, 0
    return float(df["correct"].mean()), int(len(df))

def main():
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"Start processing Scenes: {SCENES}")
    print(f"Data Root: {BASE_DATA_DIR_ROOT}")
    print(f"Missing Types to process: {MISS_TYPES}")

    for sid in SCENES:
        # 1. 动态获取当前场景的雷达坐标
        if sid not in RADAR_LLA_ALL:
            print(f"[Warning] Scene {sid} not found in RADAR_LLA_ALL dict, skipping.")
            continue
        radar_lla = RADAR_LLA_ALL[sid]

        # 2. 动态构建当前场景的数据文件夹路径
        current_scene_dir = os.path.join(BASE_DATA_DIR_ROOT, f"Scene{sid}")
        
        if not os.path.exists(current_scene_dir):
            print(f"[Warning] Directory not found: {current_scene_dir}, skipping Scene {sid}.")
            continue

        print(f"\n=== Processing Scene {sid} (LLA: {radar_lla}) ===")

        for lv in LEVELS:
            # === 新增：循环处理每种缺失类型 ===
            for miss_type in MISS_TYPES:
                
                # 构造符合 MATLAB 输出格式的文件名 pattern
                # 注意：这里使用双花括号 {{sid}} 和 {{lv}} 是为了让 f-string 保留 {sid} 和 {lv} 
                # 供 run_ttta_for_scene 内部格式化使用，而 {miss_type} 会被立即替换
                current_radar_pattern = f"scene{{sid}}_radar_{{lv}}_miss_{miss_type}.csv"
                
                try:
                    # -------------------------------------------------
                    # 运行 Baseline
                    # -------------------------------------------------
                    pairs0, C0, matches0, radar_ids0, adsb_ids0 = run_ttta_for_scene(
                        scene_id=sid, level=lv, 
                        base_dir=current_scene_dir,
                        radar_lla=radar_lla,
                        params=PARAMS_BASE,
                        use_kc=False,
                        radar_pattern=current_radar_pattern  # <--- 传入特定的缺失文件名模式
                    )

                    # -------------------------------------------------
                    # 运行 KC-DTW
                    # -------------------------------------------------
                    pairs1, C1, matches1, radar_ids1, adsb_ids1 = run_ttta_for_scene(
                        scene_id=sid, level=lv, 
                        base_dir=current_scene_dir,
                        radar_lla=radar_lla,
                        params=PARAMS_KC,
                        use_kc=True,
                        radar_pattern=current_radar_pattern  # <--- 传入特定的缺失文件名模式
                    )
                except FileNotFoundError as e:
                    print(f"[Scene {sid} Lv {lv} Miss {miss_type}] skip (missing file): {e}")
                    continue
                except Exception as e:
                    print(f"[Scene {sid} Lv {lv} Miss {miss_type}] ERROR: {e}")
                    continue

                # 保存结果
                df0 = _pairs_to_df(pairs0)
                df1 = _pairs_to_df(pairs1)

                # 文件名中加入 miss_type 以区分
                out0 = Path(RESULT_DIR) / f"scene{sid}_err{lv}_miss_{miss_type}_baseline.csv"
                out1 = Path(RESULT_DIR) / f"scene{sid}_err{lv}_miss_{miss_type}_kcdtw.csv"
                df0.to_csv(out0, index=False, encoding="utf-8-sig")
                df1.to_csv(out1, index=False, encoding="utf-8-sig")

                acc0, n0 = _safe_acc(df0)
                acc1, n1 = _safe_acc(df1)

                # 汇总信息增加 miss_type 字段
                summary_rows.append({
                    "scene": sid,
                    "level": lv,
                    "miss_type": miss_type,
                    "n_pairs": n0,
                    "acc_baseline": acc0,
                    "acc_kcdtw": acc1
                })
                print(f"[Scene {sid} Lv {lv} {miss_type}] acc {acc0:.3f}->{acc1:.3f}")

    # 保存总表
    summary = pd.DataFrame(summary_rows)
    # 调整列顺序，把 miss_type 放在 level 后面方便查看
    cols = ["scene", "level", "miss_type", "n_pairs", "acc_baseline", "acc_kcdtw"]
    summary = summary[cols]
    
    summary_path = Path(RESULT_DIR) / "summary_accuracy_all_missing_types.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nAll Done. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()