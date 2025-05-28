# 音频处理工具

本目录包含SkyrisAI的音频处理工具和相关文件。

## 必需文件

### mpv.exe

由于GitHub的文件大小限制（100MB），`mpv.exe` 文件未包含在仓库中。您需要手动下载并添加此文件：

1. 从[MPV官方网站](https://mpv.io/installation/)下载最新版本的mpv
2. 或使用此[直接链接](https://sourceforge.net/projects/mpv-player-windows/files/64bit/)下载64位Windows版本
3. 将下载的`mpv.exe`文件放在`backend/utils/`目录下

### opo.m4a

这是用于语音克隆的音频样本文件，已包含在仓库中。

## 语音克隆功能

本项目使用Minimax API进行语音克隆和文本转语音功能：

1. 上传音频样本(`opo.m4a`)进行语音克隆
2. 使用克隆的语音生成语音响应
3. MPV用于流式播放生成的音频

## 相关文件用途

- `Audio.py`: 主要音频处理工具库，包含TTS和音频处理功能
- `EM_p.py`: 情感处理相关功能
- `train.py`: 训练脚本
- `opo.m4a`: 语音克隆样本文件 