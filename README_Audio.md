
这是一个使用 `gTTS`（Google Text-to-Speech）和 `pydub` 库生成卡通风格语音的 Python 项目。该项目通过合成语音，添加背景音效（如鹦鹉鸣叫），以及一些特效处理（如颤音和语音增强），来生成有趣的音频效果。

## 功能概述

- 使用 `gTTS` 库生成中文文本语音（例如：开心、生气、平静等情绪表达）。
- 使用 `pydub` 对音频进行处理：包括添加颤音、均衡器调整、背景音效混音等。
- 可生成卡通风格的音频文件，模拟小动物语气。
- 支持自定义背景音效文件，如鹦鹉的鸣叫声。

## 依赖库

项目依赖以下 Python 库：

- `gTTS`：用于将文本转为语音。
- `pydub`：用于音频处理和效果应用。
- `numpy`：用于数值计算，特别是音频数据处理。

可以通过以下命令安装这些依赖：

```bash
pip install gTTS pydub numpy
```

## 文件说明
Audio.py：主脚本文件，包含音频生成与处理的主要代码。

new_parrot_chirp_1.wav/new_parrot_chirp_2.wav/new_parrot_chirp.wav:雪鸮原始音频（背景音）。

output_buXing.wav/output_enEn.wav/output_haoYa.wav：示例输出音频。

## 使用方法
将您需要合成的文本放入 TEXTS 列表中。例如，您可以设置 "开心", "生气", "平静" 等文本。

设置输出文件路径和背景音效文件路径：

OUTPUT_FILES 列表：用于保存每个情绪对应的输出音频文件。

PARROT_CALL_FILES 列表：提供多个背景音效文件（如鹦鹉鸣叫声）以便在生成音频时使用。

## 运行脚本：

```bash
python Audio.py
```
脚本会依次生成每个情绪的音频文件，并将它们保存在指定的路径中。音频处理过程中还会应用一些特效，如颤音、语速加快、音频增强等，最终生成卡通风格的音频文件。

## 生成的输出
每次执行脚本后，将生成以下文件（根据 TEXTS 列表中的内容）：

output_haoYa.wav：对应“开心”情绪的音频文件。

output_buXing.wav：对应“生气”情绪的音频文件。

output_enEn.wav：对应“平静”情绪的音频文件。

## 常见问题
安装问题：如果安装 pydub 或 gTTS 时遇到问题，可以尝试先安装 ffmpeg，因为 pydub 依赖 ffmpeg 处理音频文件。

```bash
pip install ffmpeg
```
音频文件生成失败：确保 gTTS 库可以正常访问 Google 服务。如果遇到网络问题，尝试重新连接。
