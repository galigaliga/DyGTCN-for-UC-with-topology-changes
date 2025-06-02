import pandas as pd
import numpy as np
import os

# ---------------------- 全局配置 ----------------------
GEN_PARAM_PATH = "./data/gen.csv"  # 机组参数文件
ADJUSTED_PATH = "./adjusted.csv"  # 调整后预测
PREDICT_PATH = "./adjusted_balance.csv"  # 原始预测
TRUTH_PATH = "./state.csv"  # 真实值
OUTPUT_DIR = "./约束违反"  # 输出目录
os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

# 加载机组参数
gen_data = pd.read_csv(GEN_PARAM_PATH, encoding='gb2312')
TG_on = gen_data.iloc[:, 6].values  # 第7列为最小运行时间
TG_off = gen_data.iloc[:, 7].values  # 第8列为最小停机时间


def check_constraints(schedule: np.ndarray, day: int) -> pd.DataFrame:
    """
    检查单日调度方案的约束满足情况
    :param schedule: 单日调度方案矩阵 (24, 54)
    :param day: 当前分析的天数 (0-32)
    :return: 包含违反约束信息的DataFrame
    """
    violations = []

    for unit in range(54):
        status = schedule[0, unit]  # 初始状态
        duration = 1

        for hour in range(1, 24):
            # 状态发生切换时触发检查
            if schedule[hour, unit] != status:
                # 检查停机持续时间是否满足最小停机时间
                if status == 0 and duration < TG_off[unit]:
                    violations.append({
                        '小时段': f"{hour - duration}-{hour - 1}",
                        '机组编号': unit + 1,
                        '状态类型': '停机',
                        '持续时间': duration,
                        '最小要求时间': TG_off[unit],
                        '约束类型': '最小停机时间不满足'
                    })
                # 检查运行持续时间是否满足最小运行时间
                elif status == 1 and duration < TG_on[unit]:
                    violations.append({
                        '小时段': f"{hour - duration}-{hour - 1}",
                        '机组编号': unit + 1,
                        '状态类型': '运行',
                        '持续时间': duration,
                        '最小要求时间': TG_on[unit],
                        '约束类型': '最小运行时间不满足'
                    })
                # 重置计数器
                status = schedule[hour, unit]
                duration = 1
            else:
                duration += 1

        # 检查最后一个时段的连续状态
        if status == 0 and duration < TG_off[unit]:
            violations.append({
                '小时段': f"{24 - duration}-23",
                '机组编号': unit + 1,
                '状态类型': '停机',
                '持续时间': duration,
                '最小要求时间': TG_off[unit],
                '约束类型': '最小停机时间不满足'
            })
        elif status == 1 and duration < TG_on[unit]:
            violations.append({
                '小时段': f"{24 - duration}-23",
                '机组编号': unit + 1,
                '状态类型': '运行',
                '持续时间': duration,
                '最小要求时间': TG_on[unit],
                '约束类型': '最小运行时间不满足'
            })

    # 转换为DataFrame并保存
    df = pd.DataFrame(violations)
    if not df.empty:
        output_path = f"{OUTPUT_DIR}/day{day + 1}/cons_unsati.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='gb18030')

    return df


def analyze_all_days():
    """分析所有56天的约束违反情况"""
    # 加载数据
    adjusted = pd.read_csv(ADJUSTED_PATH, header=None).values.astype(int)
    # adjusted = pd.read_csv(PREDICT_PATH).values.astype(int)

    adjusted_3d = adjusted.reshape(56, 24, 54)  # 重塑为(天, 小时, 机组)

    total_stats = []

    for day in range(56):
        print(f"正在分析第 {day + 1} 天...")
        daily_schedule = adjusted_3d[day]
        violations = check_constraints(daily_schedule, day)

        # 统计信息
        total_violations = len(violations)
        affected_units = violations['机组编号'].nunique() if not violations.empty else 0

        total_stats.append({
            '天': day + 1,
            '总违反点': total_violations,
            '受影响机组': affected_units
        })

    # 输出总体统计
    stats_df = pd.DataFrame(total_stats)
    stats_path = f"{OUTPUT_DIR}/约束违反统计总表.csv"
    stats_df.to_csv(stats_path, index=False, encoding='gb18030')
    print(f"总体统计结果已保存至 {stats_path}")


# 执行分析
if __name__ == "__main__":
    analyze_all_days()
