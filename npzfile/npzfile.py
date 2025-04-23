import os
import numpy as np
import pandas as pd

def convert_npz_to_single_csv(folder_path='.'):
    """
    对文件夹中每个 .npz 文件生成一个 CSV，包含所有字段：
    - 时间序列长度依据最长的一维数组
    - 一维时间序列直接作为列
    - 多维时间序列按第一维为行，展开后续维度为多列
    - 标量或静态数组按列名展开并在所有行中重复
    """
    for filename in os.listdir(folder_path):
        if not filename.endswith('.npz'):
            continue
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        # 计算时间维度为最长的一维数组长度
        lengths = [data[key].shape[0] for key in data.files
                   if isinstance(data[key], np.ndarray) and data[key].ndim >= 1]
        rows = max(lengths) if lengths else 1

        cols = {}
        for key in data.files:
            arr = data[key]
            # 标量
            if arr.ndim == 0:
                cols[key] = np.repeat(arr.item(), rows)
            # 一维且长度等于 rows，视为时间序列
            elif arr.ndim == 1 and arr.shape[0] == rows:
                cols[key] = arr
            else:
                # 如果是多维且第0维等于 rows，展开后续维度
                if arr.ndim >= 2 and arr.shape[0] == rows:
                    flat = arr.reshape(rows, -1)
                    for i in range(flat.shape[1]):
                        cols[f"{key}_{i}"] = flat[:, i]
                else:
                    # 静态一维或长度不匹配，扁平化后重复
                    flat = arr.flatten()
                    for i, v in enumerate(flat):
                        cols[f"{key}_{i}"] = np.repeat(v, rows)

        df = pd.DataFrame(cols)
        out_csv = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Converted {filename} -> {out_csv}")

if __name__ == "__main__":
    # 调用示例：当前目录
    convert_npz_to_single_csv()
