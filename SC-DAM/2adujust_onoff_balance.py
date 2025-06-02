import pandas as pd
import numpy as np
import os
import re

# --------------------------
# 初始化配置
# --------------------------
print("初始化系统配置...")
os.makedirs("./约束违反", exist_ok=True)
np.seterr(all='raise')


# --------------------------
# 数据加载与预处理
# --------------------------
def safe_load(path, dtype=None):
    df = pd.read_csv(path)
    if df.isnull().values.any():
        raise ValueError(f"检测到空值在文件 {path}")
    return df.values.astype(dtype) if dtype else df.values


print("\n加载输入数据...")

# 负荷数据
load_df = pd.read_csv("./data/new_busloads.csv")
pattern = re.compile(r'busload_.*?(\d+)$')
valid_cols = [col for col in load_df.columns
              if (m := pattern.match(col)) and 310 <= int(m.group(1)) <= 365]
valid_cols.sort(key=lambda x: int(pattern.match(x).group(1)))
test_load = load_df[valid_cols].values.T

# 机组数据
gen_df = pd.read_csv("./data/gen.csv").sort_values('gen')
min_prod = gen_df.MinProd.values.astype(np.float64)
max_prod = gen_df.MaxProd.values.astype(np.float64)
min_down = gen_df.MinTD.values.astype(int)

# 成本参数
price_df = pd.read_csv("./data/price.csv")
a, b, c = price_df.a.values, price_df.b.values, price_df.c.values

# 检修计划
print("\n加载检修计划...")
maintenance_df = pd.read_csv("./data/maintenance.csv")


def clean_maintenance_value(x):
    try:
        return int(re.sub(r'\D', '', str(x)))  # 移除非数字字符
    except:
        return 0


maintenance_df = maintenance_df.applymap(clean_maintenance_value)
maintenance_days = maintenance_df.iloc[:, 309:365].values.T


# --------------------------
# 核心约束模块
# --------------------------
def apply_maintenance_constraints(schedule):
    """优先应用检修约束"""
    maintenance_mask = np.zeros_like(schedule, dtype=bool)

    for day_idx in range(56):
        # 安全获取检修机组
        valid_units = []
        for x in maintenance_days[day_idx]:
            unit_id = int(x) - 1
            if 0 <= unit_id < 54:
                valid_units.append(unit_id)

        if valid_units:
            # 标记检修机组
            maintenance_mask[day_idx, :, valid_units] = True
            print(f"Day{310 + day_idx} 标记检修机组 {[u + 1 for u in valid_units]}")

    # 一次性关闭所有标记机组
    schedule[maintenance_mask] = 0
    return schedule, maintenance_mask


# --------------------------
# 调整算法模块
# --------------------------
def _preprocess_shutdown_units(day, schedule, maintenance_mask):
    """预处理跳过检修机组"""
    original_plan = schedule[day].copy()
    modified_plan = schedule[day].copy()
    modified = False

    for unit in range(54):
        # 跳过检修机组
        if maintenance_mask[day, 0, unit]:
            continue

        if original_plan[0, unit] == 0 and np.any(original_plan[:, unit] == 1):
            downtime = min_down[unit]
            if downtime > 0:
                start_hour = min(downtime, 23)
                if modified_plan[start_hour:, unit].sum() != (24 - start_hour):
                    modified_plan[start_hour:, unit] = 1
                    modified = True
                    print(f"Day{310 + day}机组{unit + 1}: 原计划有启动，"
                          f"强制从{start_hour}h开启（需最小停机{downtime}h）")

    if modified:
        schedule[day] = modified_plan
    return schedule

def calculate_actual_output(unit, start_hour, current_hour):
    """计算机组从start_hour到current_hour的有效出力"""
    running_hours = current_hour - start_hour + 1
    theory_p = min_prod[unit] * running_hours
    return min(theory_p, max_prod[unit])

def basic_economic_adjust(status, required_gap, schedule, current_hour, maintenance_status):
    """经济调整排除检修机组"""
    if required_gap <= 1e-6:
        return schedule

    # 筛选可用机组
    available = np.logical_and(status == 0, ~maintenance_status)
    offline = np.where(available)[0]

    if not offline.size:
        return schedule

    cost = a[offline] * min_prod[offline] + b[offline] + c[offline] / min_prod[offline]
    sorted_units = offline[np.argsort(cost)]

    cumulative = 0.0
    for unit in sorted_units:
        schedule[current_hour:, unit] = 1
        cumulative += min_prod[unit]
        if cumulative >= required_gap:
            break
    return schedule


def backtrack_ramp_adjust(gap, schedule, current_hour, maintenance_status):
    """回溯调整排除检修机组"""
    candidate_list = []
    for unit in range(54):
        # 过滤检修机组（新增判断）
        if maintenance_status[unit] or schedule[current_hour, unit] == 1:
            continue

        for lookback in range(current_hour + 1):
            start = current_hour - lookback
            if np.any(schedule[start:current_hour + 1, unit] == 1):
                continue

            possible_p = calculate_actual_output(unit, start, current_hour)
            if possible_p < 1e-6:
                continue

            unit_cost = a[unit] * min_prod[unit] + b[unit] + c[unit] / min_prod[unit]
            candidate_list.append((unit_cost, possible_p, lookback, unit, start))

    if not candidate_list:
        return 0.0, schedule

    sorted_candidates = sorted(candidate_list, key=lambda x: (-x[1], x[0], x[2]))
    filled = 0.0
    modified = schedule.copy()
    for _, p, _, unit, start in sorted_candidates:
        if filled >= gap:
            break
        modified[start:current_hour + 1, unit] = 1
        actual_add = min(p, gap - filled)
        filled += actual_add
        print(f"  ↖ 回溯启动机组 #{unit + 1} 于时段{start}[提前{current_hour - start}h] "
              f"贡献 {actual_add:.2f}/{p:.2f} MW")
    return filled, modified


def advanced_adjustment(status, load, schedule, hour, maintenance_status):
    """调整主逻辑集成检修状态"""
    current_power = np.dot(status, max_prod)
    gap = load - current_power

    stage1 = basic_economic_adjust(status, gap, schedule.copy(), hour, maintenance_status)
    stage1_power = np.dot(stage1[hour], max_prod)
    residual = load - stage1_power

    if residual <= 1e-6:
        return stage1

    filled, stage2 = backtrack_ramp_adjust(residual, stage1, hour, maintenance_status)
    final_power = np.dot(stage2[hour], max_prod)
    if final_power < load - 1e-6:
        print(f"⚠️ 无法完全填补缺口！剩余 {load - final_power:.2f} MW")
    return stage2


# --------------------------
# 主处理流程
# --------------------------
print("\n启动主调整流程...")
adjusted_3d = safe_load("./data/all_predict_value.csv", int).reshape(56, 24, 54)

# 优先应用检修约束并获取掩码
adjusted_3d, maintenance_mask = apply_maintenance_constraints(adjusted_3d)

debug_log = open("./adjustment_log.txt", "w")

# 预处理时传入检修掩码
for day_idx in range(56):
    adjusted_3d = _preprocess_shutdown_units(day_idx, adjusted_3d, maintenance_mask)

for day_idx in range(56):
    current_day = 310 + day_idx
    print(f"\n处理第 {current_day} 天:")
    daily_schedule = adjusted_3d[day_idx].copy()
    daily_load = test_load[day_idx]

    # 获取当日检修状态
    day_maintenance = maintenance_mask[day_idx].any(axis=0)

    for hour in range(24):
        current_load = daily_load[hour]
        initial_status = daily_schedule[hour].copy()
        initial_power = np.dot(initial_status, max_prod)

        if initial_power >= current_load - 1e-6:
            continue

        original_gap = current_load - initial_power
        print(f"  ⌚ 时段 {hour}: 初始缺口 {original_gap:.2f} MW")
        debug_log.write(f"Day{current_day} Hour{hour}: 初始缺口 {original_gap:.2f}\n")

        updated_schedule = advanced_adjustment(
            initial_status, current_load,
            daily_schedule.copy(), hour,
            day_maintenance  # 传入当日检修状态
        )
        daily_schedule = updated_schedule
        adjusted_3d[day_idx] = daily_schedule

        final_status = daily_schedule[hour]
        final_power = np.dot(final_status, max_prod)
        debug_log.write(f"Day{current_day} Hour{hour}: 最终出力 {final_power:.2f}\n")

# 最终确认检修状态（防止后续修改）
adjusted_3d[maintenance_mask] = 0
debug_log.close()

# --------------------------
# 结果保存与验证
# --------------------------
print("\n保存调整结果...")
adjusted_2d = adjusted_3d.reshape(56, -1)
pd.DataFrame(adjusted_2d).to_csv("./adjusted_balance.csv", header=False, index=False)

def comprehensive_validation():
    """最终验证（增强版）"""
    print("\n启动最终验证...")
    saved_data = pd.read_csv("./adjusted_balance.csv", header=None).values
    saved_3d = saved_data.reshape(56, 24, 54)

    # 供电缺口验证
    violations = []
    for day in range(56):
        day_loads = test_load[day]
        for hour in range(24):
            status = saved_3d[day, hour]
            total_p = np.dot(status, max_prod)
            if total_p < day_loads[hour] - 1e-6:
                gap = day_loads[hour] - total_p
                violations.append((310 + day, hour, gap))

    # 检修约束验证
    maintenance_violations = []
    for day in range(56):
        units = [int(x)-1 for x in maintenance_days[day] if x > 0]
        for unit in units:
            if np.any(saved_3d[day, :, unit] != 0):
                maintenance_violations.append((310+day, unit+1))

    # 生成报告
    if violations:
        report_df = pd.DataFrame(violations, columns=["Day", "Hour", "Gap"])
        report_df.to_csv("./约束违反/final_gaps.csv", index=False)
        print(f"❌ 发现 {len(violations)} 个供电缺口")

        top_gaps = report_df[report_df.Gap > 10].sort_values("Gap", ascending=False)
        if not top_gaps.empty:
            print("\n严重缺口案例：")
            print(top_gaps.head().to_string(index=False))

    if maintenance_violations:
        print(f"❌ 发现 {len(maintenance_violations)} 处检修约束违反")
        pd.DataFrame(maintenance_violations,
                   columns=["Day", "Unit"]).to_csv("./约束违反/maintenance_violations.csv", index=False)

    if not violations and not maintenance_violations:
        print("✅ 所有约束检查通过")

comprehensive_validation()
print("\n处理流程全部完成！")
