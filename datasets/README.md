# Datasets

该目录用于存放论文实验使用的本地数据。

当前推荐通过以下脚本自动构建开放文化图像数据集：

```bash
python data_tools/download_wikimedia_commons.py --output-root datasets/wikimedia_cultural_patterns --max-per-category 20
```

默认会下载以下三类开放文化图像：

- blue_and_white_porcelain
- gu_embroidery
- paper_cutting

下载完成后会生成：

- `images/`
- `manifest.jsonl`
- `metadata.json`

其中 `manifest.jsonl` 可直接作为批量实验输入清单使用。

## 已准备好的本地起始数据

当前已在本地生成一份可直接用于原型实验的起始数据集：

- `datasets/starter_cultural_patterns/`

该目录当前包含：

- 4 张青花瓷相关图像
- 1 张顾绣相关图像
- 2 张剪纸相关图像

对应清单文件：

- `datasets/starter_cultural_patterns/manifest.jsonl`
- `datasets/starter_cultural_patterns/metadata.json`
- `datasets/starter_cultural_patterns/download_report.json`
- `datasets/starter_cultural_patterns/image_stats.json`

推荐直接配合以下配置文件执行起始实验：

- `configs/starter_experiment_grid.json`

对应批量实验命令：

```bash
python experiments/run_grid.py --config configs/starter_experiment_grid.json --output-root batch_outputs/starter --device cuda
```

如需继续扩充样本规模，优先运行：

```bash
python data_tools/download_wikimedia_commons.py --output-root datasets/wikimedia_cultural_patterns --max-per-category 20
```
