# Sarkaz Translator

一个面向中文文本与“萨卡兹/游戏字母串”之间转换的本地工具集，包含：

- **解码**：把经过 `Unicode % 56 -> 字母` 映射后的英文字符串还原为中文候选句。
- **编码**：把中文文本编码成对应的英文字母串。
- **可视化**：用 Gradio 提供本地 Web 界面，便于直接编码/解码。

项目当前实现的核心思路是：

1. 根据固定映射表构建 `letter -> 候选字符列表` 的反向索引。
2. 使用字频、二元/三元/四元统计与转移矩阵进行 **Beam Search** 粗筛。
3. 可选地调用 **LLM** 对候选结果做自然度重排。
4. 最终对候选句重新编码校验，确保结果与原始输入一致。

## 功能特性

- 支持 **中文 / 中文标点 / 数字** 编码
- 支持命令行解码与交互式解码
- 支持本地 **Gradio WebUI**
- 支持可选 **LLM 重排**（提升候选句自然度）
- 内置 `EndfieldFonts/` 字体 PNG，可把编码结果渲染成图片
- 附带 `scripts/` 数据处理脚本，可重新生成语料统计数据

## 项目结构

```text
.
├─ app.py                  # Gradio WebUI 入口（编码 / 解码）
├─ main.py                 # 命令行解码入口
├─ requirements.txt        # 依赖列表
├─ src/
│  ├─ mapping.py           # 编码映射表与反向索引
│  ├─ encoder.py           # 编码器与结果校验
│  ├─ decoder.py           # 解码主流程编排
│  ├─ beam_search.py       # Beam Search 实现
│  ├─ frequency.py         # 频率数据加载与候选排序
│  ├─ llm_scorer.py        # LLM 打分 / 重排
│  ├─ ocr.py               # 模板匹配 OCR 工具函数
│  └─ config.py            # 搜索参数与默认 LLM 配置
├─ data/
│  ├─ game_unigram_freq.json
│  ├─ game_bigram_freq.json
│  ├─ game_trigram_freq.json
│  ├─ game_fourgram_freq.json
│  └─ game_transition.json
├─ EndfieldFonts/          # 字母 PNG 素材
├─ scripts/                # 数据构建脚本
└─ docs/                   # 设计文档与方案记录
```

## 工作原理

### 1. 编码

对每个字符执行：

```text
unicode_codepoint % 56 -> remainder -> letter
```

余数到字母的映射写在 `src/mapping.py` 的 `FORWARD_MAP` 中。

示例：

```python
from src.encoder import encode
print(encode("你好，世界123"))
# ytqhfpjz
```

### 2. 解码

由于编码是多对一映射，单个字母通常会对应数百到上千个候选字符，因此不能直接逆推。项目采用以下策略：

- 先根据余数反查出每个位置的候选字符集合
- 用字频过滤掉低优先级候选（`candidate_k`）
- 用 Beam Search 按上下文概率保留最优路径（`beam_width`）
- 可选地让 LLM 对候选句做自然度重排
- 最后重新编码验证，剔除不匹配结果

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `openai`
- `opencv-python`
- `numpy`
- `gradio`

### 2. 配置 LLM（可选）

如果你需要 **命令行解码时自动使用 LLM 重排**，请检查 `src/config.py` 中的以下配置：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_TEMPERATURE`

> 建议只在本地保存自己的密钥，不要把真实密钥提交到版本库。

如果你 **不想在命令行中使用 LLM**，可以把 `LLM_API_KEY` 设为空字符串。

WebUI 中则可以通过界面勾选“启用 LLM 打分”后再填写 Key / Base URL / Model，无需修改代码。

## 使用方法

### 命令行解码

#### 单次解码

```bash
python main.py ytqhfpjz
```

#### 交互模式

```bash
python main.py -i
```

交互模式下输入编码串，输入 `q` 退出。

#### 查看帮助

```bash
python main.py -h
```

### 启动 WebUI

```bash
python app.py
```

启动后会在本地打开/输出一个 Gradio 地址。当前界面提供：

- **Decode**：输入编码后的字母串，返回候选中文句
- **Encode**：输入中文文本，返回编码串与对应字母图片

#### WebUI 可调参数

解码页高级设置中可以调整：

- `BEAM_WIDTH`（默认 2000）
- `CANDIDATE_K`（默认 200）
- `TOP_N`（默认 5）
- `TOP_BEAM_RESULTS`（默认 50）
- `LAMBDA_UNIGRAM`（默认 0.3）
- `LAMBDA_TRIGRAM`（默认 0.5）
- `LAMBDA_FOURGRAM`（默认 0.8）

## 作为 Python 模块使用

### 编码

```python
from src.encoder import encode

encoded = encode("你好，世界123")
print(encoded)
```

### 解码

```python
from src.mapping import build_reverse_index
from src.frequency import load_freq
from src.decoder import decode

reverse_index = build_reverse_index()
unigram = load_freq("data/game_unigram_freq.json")
bigram = load_freq("data/game_bigram_freq.json")
transition = load_freq("data/game_transition.json")
trigram = load_freq("data/game_trigram_freq.json")
fourgram = load_freq("data/game_fourgram_freq.json")

results = decode(
    "ytqhfpjz",
    reverse_index,
    unigram,
    bigram,
    api_key=None,              # 不启用 LLM
    llm_model=None,
    llm_base_url=None,
    transition=transition,
    trigram_freq=trigram,
    fourgram_freq=fourgram,
)

for score, sentence in results:
    print(score, sentence)
```

## 数据说明

项目已自带解码所需的统计数据：

- `game_unigram_freq.json`：单字频率
- `game_bigram_freq.json`：二元组频率
- `game_trigram_freq.json`：三元组频率
- `game_fourgram_freq.json`：四元组频率
- `game_transition.json`：字符转移矩阵

如果你希望重新构建或更新数据，可参考：

- `scripts/extract_cn_texts.py`
- `scripts/build_frequency_data.py`
- `scripts/build_ngram_stats.py`

## 已知限制

- 该编码方式本身 **不可逆**，解码结果是“候选句排序”，不是数学上的唯一还原。
- 句子越长、上下文越稀疏，候选空间越大，解码难度越高。
- 不启用 LLM 时，结果更依赖本地统计数据质量。
- 当前命令行入口主要提供 **解码**；编码更适合通过 WebUI 或直接调用 `src.encoder.encode()`。
- `src/ocr.py` 已提供 OCR 基础能力，但当前 WebUI 版本尚未暴露截图识别入口。

## 开发与调试建议

- 调整搜索效果时，优先观察 `src/config.py` 中的 Beam / N-gram 参数
- 修改映射规则时，同时检查 `src/mapping.py` 与 `src/encoder.py`
- 如果结果质量下降，优先检查 `data/` 中频率数据是否匹配当前语料
- `decoder.py` 会输出部分调试信息到标准错误，便于观察 Beam Search 过程

## 参考文档

- `docs/superpowers/specs/2026-04-21-sarkaz-decoder-design.md`
- `docs/superpowers/specs/2026-04-21-webui-design.md`

如果你只是想快速体验，最短路径是：

```bash
pip install -r requirements.txt
python app.py
```
