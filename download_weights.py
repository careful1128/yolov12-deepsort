import os
from ultralytics import YOLO

# 所有要确保存在的模型
models = [
    "yolov12n.pt",
    "yolov12s.pt",
    "yolov12m.pt",
    "yolov12l.pt",
    "yolov12x.pt",
]

# 确保有 weights 目录
os.makedirs("weights", exist_ok=True)

for model_name in models:
    local_path = os.path.join("weights", model_name)
    if not os.path.exists(local_path):
        print(f"➡️ {model_name} 不在 weights/ 中，开始下载到本地 ...")
        # 用 YOLO 下载到缓存目录
        model = YOLO(model_name)
        # 再把缓存文件复制到 weights/
        cache_path = model.ckpt_path
        os.system(f"cp '{cache_path}' '{local_path}'")
        print(f"✅ 已复制 {model_name} 到 weights/")
    else:
        print(f"✔️ 已存在: {local_path}")
