from pathlib import Path
import torch

path = Path(r"E:\Projects\曦源计划\数据\SpectraDB\raw\qm9s\qm9s.pt")

print(f"文件：{path}")
print(f"大小：{path.stat().st_size / 1024**3:.3f} GB")
print("正在载入，可能需要一段时间……")

obj = torch.load(path, map_location="cpu", weights_only=False)

print("\n顶层对象类型：", type(obj))

if isinstance(obj, dict):
    print("字典键：", list(obj.keys()))

    for key, value in obj.items():
        shape = getattr(value, "shape", None)
        try:
            length = len(value)
        except TypeError:
            length = None

        print(
            f"{key}: type={type(value)}, "
            f"shape={shape}, len={length}"
        )

elif isinstance(obj, (list, tuple)):
    print("样本数：", len(obj))

    first = obj[0]
    print("首个样本类型：", type(first))

    keys_attr = getattr(first, "keys", None)
    if callable(keys_attr):
        keys = list(keys_attr())
        print("首个样本字段：", keys)

        for key in keys:
            value = getattr(first, key)
            print(
                f"{key}: type={type(value)}, "
                f"shape={getattr(value, 'shape', None)}"
            )
    else:
        print("首个样本：", first)

else:
    print("对象摘要：", obj)
