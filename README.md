# CLIP 图文检索与 C++ 推理部署

## 这个项目做什么

用户输入一张图片，系统从 COCO 的 25014 条描述中找出最匹配的前 5 条。

项目包含两条路径：

- 快速模式：CLIP + C++ / ONNX Runtime，延迟低
- 高精度模式：CLIP 召回 Top-20，再用 BLIP 重新排序，准确率高

## 核心结果

### 完整 COCO baseline

数据：COCO 2017 val2017，5000 张图片，25014 条描述。

| 方法 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| CLIP baseline | 0.5002 | 0.7500 | 0.8324 |

### 固定 1000 张测试集上的改进

| 方法 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| CLIP baseline | 0.7020 | 0.8980 | 0.9520 |
| CLIP Top-20 + BLIP rerank | 0.9100 | 0.9680 | 0.9770 |

Recall@1 提升：**20.80 个百分点**。

### 延迟

| 模式 | 平均延迟 |
|---|---:|
| C++ 快速模式，CLIP 端到端 Top-K | 约 32 ms |
| Python ONNX 快速模式 | 约 36 ms |
| BLIP 重排序 | 约 1.3 s/张 |

优化后的详细延迟结果：`results/cpp_end_to_end_latency_optimized.csv`

C++ 与 Python 输出一致性：

- 图片向量余弦相似度：`1.0`
- 最大绝对误差：`5.55e-07`
- Top-5 结果一致

## 系统流程

```text
快速模式：
图片
→ OpenCV 预处理
→ ONNX CLIP 图片编码
→ 和 25014 条文字向量计算相似度
→ 输出 Top-5 描述
```

```text
高精度模式：
图片
→ CLIP 召回 Top-20
→ BLIP ITM 重排序
→ 输出 Top-5 描述
```

## 项目结构

```text
├── checkpoints/      ONNX 模型和测试输入
├── cpp/              C++ 推理源码和构建配置
├── data/             COCO metadata 和文字向量索引
├── notebooks/        实验 Notebook
├── results/          指标、错误案例、延迟结果
└── README.md
```


## 仓库未包含的文件

为控制仓库体积并避免分发第三方权重，以下文件未提交：

- `checkpoints/clip_image_encoder.onnx` 及外部权重文件
- `data/text_index/text_features.bin`
- `results/*.pt` 向量缓存

复现时：

1. 按 `notebooks/05_coco5000_final_baseline.ipynb` 生成 COCO 图片与文字特征。
2. 按 `notebooks/08_onnx_export.ipynb` 导出 CLIP 图片编码器 ONNX。
3. 按 `cpp/export_text_assets.ipynb` 生成 C++ 检索所需的文字向量索引。
4. 将 COCO 图片、ONNX 模型和文字索引放到 README 中对应路径，再运行 C++ 推理。

## 怎么运行 C++ 版本

环境：macOS Apple Silicon、OpenCV 5、ONNX Runtime、CMake。

```bash
cd cpp
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j4

./build/clip_infer \
  ../checkpoints/clip_image_encoder.onnx \
  /path/to/image.jpg \
  output.bin \
  ../data/text_index
```

输出包括：

```text
Top-5 描述
图片读取/预处理/模型/Top-K 各阶段延迟
端到端延迟
```

## 我做了什么

- 构建 COCO 图文检索数据与评测流程
- 完成 CLIP baseline 和错误分析
- 设计 CLIP Top-20 + BLIP 重排序流程
- 完成 Top-K 消融和延迟测试
- 将图片编码器导出 ONNX
- 用 C++ / ONNX Runtime 完成端到端 Top-K 检索
- 用 Apple Accelerate 优化 C++ Top-K
- 验证 Python/C++ 输出一致性

## 局限

- BLIP 重排序准确率高，但延迟约 1.3 秒
- 完整 5000 张图片完成了 CLIP baseline，BLIP 重排序在固定 1000 张测试集上评测
- C++ 当前只部署 CLIP 快速模式，BLIP 重排序仍在 Python 侧
- 尚未接入 FAISS、向量量化或 GPU 推理

## 详细实验资料

- `COCO5000错误分类分析.docx`：100 个错误样本分类
- `notebooks/05_coco5000_final_baseline.ipynb`：完整 baseline
- `notebooks/07_reranking.ipynb`：BLIP 重排序
- `notebooks/08_onnx_export.ipynb`：ONNX 导出
- `results/`：全部指标和错误案例

## 开源与致谢

使用 OpenCLIP、CLIP、BLIP、COCO、ONNX Runtime 和 OpenCV。

重排序模型使用 `Salesforce/blip-itm-base-coco` 预训练权重。本项目的主要工作是数据与评测流程、失败分析、两阶段检索设计、ONNX/C++ 部署和延迟优化。
