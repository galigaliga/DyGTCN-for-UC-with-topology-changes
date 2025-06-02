import pandas as pd
import numpy as np
import os
import re  # 新增正则表达式模块

# 确保输出目录存在
os.makedirs("./调整后约束违反", exist_ok=True)

# 加载负荷数据
load_df = pd.read_csv("./data/new_busloads.csv")

# 稳健的列筛选逻辑
test_columns = []
for col in load_df.columns:
    if not col.startswith("busload_"):
        continue
    # 使用正则表达式提取列名末尾的数字
    match = re.search(r'(\d+)$', col)
    if not match:
        continue
    day_num = int(match.group(1))
    if 310 <= day_num <= 365:
        test_columns.append(col)
# 按数值排序保证顺序正确
test_columns = sorted(test_columns,
                      key=lambda x: int(re.search(r'(\d+)$', x).group(1)))

test_load = load_df[test_columns].values.T  # 转置后形状(56天, 24小时)

# 加载机组数据
gen_df = pd.read_csv("./data/gen.csv")
gen_df = gen_df.sort_values("gen")  # 确保机组按gen排序
max_prod = gen_df["MaxProd"].values  # shape (54,)

# 加载预测数据
PREDICT_PATH = "./adjusted_balance.csv"
adjusted = pd.read_csv(PREDICT_PATH,header=None).values.astype(int)
adjusted_3d = adjusted.reshape(56, 24, 54)  # (天, 小时, 机组)

violations = []
for day_idx in range(56):
    actual_day_number = 310 + day_idx  # 校准实际日历天数

    for hour_idx in range(24):
        online_units = adjusted_3d[day_idx, hour_idx, :]
        max_prod_sum = np.dot(online_units, max_prod)
        load_value = test_load[day_idx, hour_idx]

        if max_prod_sum < load_value:
            violations.append([
                actual_day_number,
                hour_idx,
                max_prod_sum - load_value
            ])

# 存储结果
result_df = pd.DataFrame(
    violations,
    columns=["天数", "时段（0-23）", "供电缺口"]
)
result_df.to_csv("./调整后约束违反/balance.csv", index=False, encoding='gb18030')

print(f"处理完成，发现{len(violations)}处供电缺口，结果路径：./调整后约束违反/balance.csv")
