"""
import pickle

def check_pkl_file(pkl_path, check_uid_only=False, max_print=5):
    print(f"🔍 正在加载: {pkl_path}")
    try:
        data = pickle.load(open(pkl_path, "rb"))
    except Exception as e:
        print(f"[错误] 读取失败: {e}")
        return

    print(f"✅ 成功加载，字段列表: {list(data.keys())}")

    total = len(next(iter(data.values())))
    print(f"📦 样本数量: {total}")

    for key in data:
        field = data[key]
        print(f"字段 `{key}` 类型: {type(field)}, 长度: {len(field)}")

    # 检查 uid 合法性
    if "uid" in data:
        uid_list = data["uid"]
        print(f"\n🔍 正在检查前 {max_print} 个 uid:")
        for i, uid in enumerate(uid_list[:max_print]):
            print(f"  uid[{i}] = {uid}, type: {type(uid)}")
        non_ints = [i for i, u in enumerate(uid_list) if not isinstance(u, int)]
        if non_ints:
            print(f"❌ 错误: 有 {len(non_ints)} 个 uid 不是 int，例如位置: {non_ints[:10]}")
        else:
            print("✅ 所有 uid 都是 int")

    if not check_uid_only:
        print("\n🔍 正在检查所有序列字段是否等长（前5条样本）:")
        keys = [k for k in data if isinstance(data[k], list) and isinstance(data[k][0], list)]
        for key in keys:
            print(f"字段 `{key}`:")
            for i in range(min(max_print, total)):
                print(f"  len({key}[{i}]) = {len(data[key][i])}")

    print("\n✅ 检查完成")

# ✅ 示例调用（改成你的路径）
check_pkl_file("D:/postgraduate/pykt/pykt/data/assist2015/train_valid_sequences.csv_0.pkl")
"""

import pandas as pd

# 替换为你的文件路径

import pandas as pd

# 替换为你的 CSV 文件路径
file_path = 'D:\\postgraduate\\pykt\\pykt\\data\\assist2015\\train_valid_sequences.csv'

# 读取前5行数据
df = pd.read_csv(file_path, nrows=5)

# 打印前5行数据内容
print("🔹前5行数据内容：")
print(df)

# 打印所有列名（特征标签）
print("\n🔸特征标签（列名）：")
print(df.columns.tolist())

# 检查是否包含 'uid' 特征
if 'uid' in df.columns:
    print("\n✅ 存在 'uid' 特征列")
else:
    print("\n❌ 不存在 'uid' 特征列")

df.to_csv('D:\\Documents\\Desktop\\akt_print_data\\first_5_rows.csv', index=False)

print("✅ 前5行内容已保存为 'first_5_rows.csv'")