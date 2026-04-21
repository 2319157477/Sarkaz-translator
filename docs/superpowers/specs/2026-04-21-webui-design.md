# Sarkaz Decoder WebUI 设计文档

## 概述

为 Sarkaz Decoder 项目构建基于 Gradio 的本地 WebUI，提供 Decode（解码）和 Encode（编码）两个功能页面。

## 技术选型

- **框架**: Gradio（gr.Blocks + gr.Tabs）
- **架构**: 单文件 `app.py`
- **运行方式**: 本地 `python app.py`
- **依赖**: 复用现有 `src/` 模块，新增 `gradio` 依赖

## 数据加载

启动时预加载所有数据文件，存为模块级变量，避免每次请求重复加载：
- `reverse_index`（build_reverse_index）
- `unigram`、`bigram`、`trigram`、`fourgram`（load_freq）
- `transition`（load_freq）
- OCR SVG 模板（render_templates）

## Decode 标签页

### 输入方式

两种输入方式，用嵌套 `gr.Tab` 切换：

1. **直接输入**: `gr.Textbox(lines=3)`，支持多行输入
2. **截图识别**:
   - 用户上传截图 → `gr.ImageEditor`（crop 模式）框选文字区域
   - 框选区域可能包含多行文本
   - 裁剪后的图片传给 `ocr.py` 的函数识别
   - 识别结果（多行）自动填入文本框

OCR 参数：height 从框选区域的裁剪高度自动推算（基于裁剪区域高度与预估行数的比值），threshold 在高级设置中由用户调整。

### LLM 打分（可选）

- `gr.Checkbox` 开关，默认关闭
- 启用后展开配置区域：API Key、Base URL、Model 三个文本框
- 启用但配置为空时，阻止提交并提示填写

### 解码流程

1. 获取文本框内容，按 `\n` 拆分为多行
2. 逐行调用 `decode()`
3. 结果按行分组展示：每行显示原始编码 + 对应的 top-N 候选表格（排名、LLM 分数、候选句子）

### 高级设置

`gr.Accordion` 折叠区域，包含：
- BEAM_WIDTH（默认 2000）
- CANDIDATE_K（默认 200）
- TOP_N（默认 5）
- TOP_BEAM_RESULTS（默认 50）
- LAMBDA_UNIGRAM（默认 0.3）
- LAMBDA_TRIGRAM（默认 0.5）
- LAMBDA_FOURGRAM（默认 0.8）
- OCR threshold（默认 0.7）

## Encode 标签页

### 输入

`gr.Textbox`，用户输入中文语句。

### 编码流程

1. 调用 `encoder.encode()` 得到英文字母串
2. 用 `EndfieldFonts/` 中的 SVG 模板，按字母逐个用 `cairosvg` 渲染为 PNG，再用 numpy/cv2 水平拼接成一张图片

### 输出

- 编码结果文本（可复制）
- 拼接后的游戏字符图片（可下载）

## 错误处理

- OCR 识别不到字符 → 提示"未识别到文字，请调整框选区域或 threshold"
- 解码某行无候选（reverse_index 查不到字母）→ 跳过该行并提示
- LLM 打分启用但 API 配置为空 → 阻止提交并提示填写
- Encode 输入包含非 CJK/标点/数字字符 → 提示哪些字符无法编码

## 文件结构

```
app.py              # WebUI 入口，单文件 Gradio 应用
requirements.txt    # 新增 gradio 依赖
```

不新增其他文件，所有 WebUI 逻辑在 `app.py` 中完成，调用现有 `src/` 模块。
