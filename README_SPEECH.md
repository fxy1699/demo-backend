# 文本转语音功能使用说明

本项目集成了卡通风格的文本转语音功能，可以将大模型生成的文本转换为带有卡通音效的语音输出。

## 功能特点

- **卡通风格**：带有鹦鹉音效的卡通风格语音，适合有趣互动场景
- **普通人声**：标准中文语音合成，适合正式场景
- **长文本支持**：自动拆分长文本并合成连贯的语音输出
- **灵活输出**：支持Base64和文件两种输出方式

## API 接口

### 1. 独立的文本转语音接口

```
POST /api/text-to-speech
```

请求体：
```json
{
  "text": "要转换的文本内容",
  "style": "cartoon",  // 可选: cartoon(卡通风格), normal(普通人声)
  "return_type": "base64"  // 可选: base64, file
}
```

响应示例（base64模式）：
```json
{
  "audio": "base64编码的音频数据..."
}
```

file模式会直接返回音频文件。

### 2. 大模型生成带语音输出

```
POST /api/generate-text
```

请求体：
```json
{
  "prompt": "用户提问内容",
  "system_prompt": "系统提示（可选）",
  "with_audio": true,  // 是否生成语音
  "audio_style": "cartoon"  // 语音风格，可选cartoon或normal
}
```

响应示例：
```json
{
  "text": "大模型生成的文本回复",
  "audio": "base64编码的音频数据...",
  "model": "使用的模型名称",
  "source": "api"
}
```

### 3. 多模态生成带语音输出

```
POST /api/generate-multimodal
```

使用表单数据（multipart/form-data）上传，额外增加以下参数：

- `with_audio`：设置为`true`启用语音输出
- `audio_style`：语音风格，可选`cartoon`或`normal`

## 前端使用示例

在前端可以这样使用：

```javascript
// 播放语音
if (result.audio) {
  const audio = new Audio(`data:audio/wav;base64,${result.audio}`);
  audio.play();
}

// 下载语音
function downloadAudio(audioBase64) {
  const link = document.createElement('a');
  link.href = `data:audio/wav;base64,${audioBase64}`;
  link.download = '语音回复.wav';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
```

## 注意事项

1. 文本长度限制为1000字符，超过会被拒绝
2. 卡通风格语音使用了鹦鹉音效和特殊处理，使声音听起来像动画角色
3. 长文本会自动拆分为句子单独处理，然后合并，可能需要较长处理时间
4. 语音生成依赖Google文本转语音服务(gTTS)，需要网络连接

## 排障指南

如果遇到语音生成失败，可能的原因：

1. **gTTS网络错误**：确保服务器能访问Google服务
2. **音频文件生成失败**：检查服务器临时文件目录权限
3. **背景音文件缺失**：确认`utils/audio_files`目录中包含了所有必要的背景音文件

## 自定义设置

如需自定义语音效果，可以修改`Audio.py`中的以下参数：

- `apply_tremolo`函数的`depth`和`freq`参数控制颤音效果
- `process_audio`函数中的`playback_speed`和`frame_rate`控制语速和音调
- `PARROT_CALL_FILES`列表可以添加或替换不同的背景音效文件 