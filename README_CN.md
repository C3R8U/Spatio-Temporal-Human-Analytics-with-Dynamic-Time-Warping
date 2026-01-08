# STHA Sports Analytics（可复现材料包）

本仓库提供面向可复现的参考实现，覆盖两类体育场景指标计算：
- 转体角估计：躯干法向量在水平面投影的有符号夹角累积
- 体操评分：动态时间规整 Dynamic Time Warping（DTW）对齐后的骨架差异作为扣分量，并映射为 10 分制得分

包含：
- 核心算法（src/stha/）
- 预处理脚本（scripts/）
- 评分/角度脚本（scripts/）
- 脱敏示例骨架（data/sample/）
- 划分文件示例（splits/）

说明：data/sample/ 为合成/脱敏示例数据，用于验证流程是否可跑通。真实实验请替换为你的视频关键点/骨架提取结果。

## 安装
```bash
python -m venv .venv
source .venv/bin/activate   # Windows：.venv\Scripts\activate
pip install -r requirements.txt
```

## 数据格式
CSV 列：frame,joint,x,y,z（每行对应某一帧某一关节）

## 快速运行
转体角示例：
```bash
python scripts/evaluate_rotation.py --cfg configs/default.yaml --input data/sample/rotation_demo.csv
```

体操评分示例：
```bash
python scripts/score_gymnastics.py --cfg configs/default.yaml \
  --test data/sample/gym_test.csv --template data/sample/gym_template.csv
```
