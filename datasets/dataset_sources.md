# Open Dataset Sources for Cultural Pattern Experiments

当前优先选择以下开放文化图像来源，用于构建论文实验数据：

## 1. Blue and White Porcelain

- Source: Wikimedia Commons
- Category: `Blue and white porcelain of China`
- URL: https://commons.wikimedia.org/wiki/Category:Blue_and_white_porcelain_of_China
- Suggested prompt: `blue and white porcelain decorative pattern, Chinese ceramic motif, high detail`

## 2. Gu Embroidery

- Source: Wikimedia Commons
- Category: `Gu embroidery`
- URL: https://commons.wikimedia.org/wiki/Category:Gu_embroidery
- Suggested prompt: `traditional Chinese embroidery pattern, silk textile motif, ornate and detailed`

## 3. Chinese Paper Cutting

- Source: Wikimedia Commons
- Category: `Paper cutting`
- URL: https://commons.wikimedia.org/wiki/Category:Paper_cutting
- Related Chinese subcategory exists under this category tree.
- Suggested prompt: `traditional Chinese paper-cut pattern, folk art motif, symmetric and detailed`

## Local Downloader

项目已提供自动下载脚本：

```bash
python data_tools/download_wikimedia_commons.py --output-root datasets/wikimedia_cultural_patterns --max-per-category 20
```

脚本默认会生成：

- `datasets/wikimedia_cultural_patterns/images/`
- `datasets/wikimedia_cultural_patterns/manifest.jsonl`
- `datasets/wikimedia_cultural_patterns/metadata.json`
- `datasets/wikimedia_cultural_patterns/download_report.json`

## Current Environment Note

当前工作环境的终端外网下载会出现连接超时，因此已完成本地下载器与数据源配置，但批量图片抓取需要在网络可访问的环境中执行。
