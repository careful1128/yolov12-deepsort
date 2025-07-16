# 项目简介：
使用YOLOv12+Deepsort实现目标检测和追踪计数。

## Export
```python
from ultralytics import YOLO

detector = YOLO('weights/yolov12x.pt')

```

## DeepSORT追踪器

```python
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30, n_init=3)

```

## 运行 Demo

```
python demo.py
```

## yolo v12 中加入attention机制与之前 unet 中 attention 模块的区别

![alt text](image.png)

## 在 YOLOv12 Area Attention 中：

```python
Input: X (B×C×H×W)

1. 将 X 切成多个 block
   => [B, num_blocks, block_tokens, C]

2. 在每个 block 内做线性映射：
   Q_block = X_block @ W^Q
   K_block = X_block @ W^K
   V_block = X_block @ W^V

3. 在每个 block 内计算 attention：
   Softmax(Q_block * K_block^T / sqrt(d)) @ V_block

4. 将所有 block 的输出 reshape 回原图
```

