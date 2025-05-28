# 大模型集成说明

本项目支持使用通义千问2.5-Omni大模型进行更准确的情绪分析和文本生成。支持通过在线API调用或本地模型两种方式。

## 功能介绍

- 通过LLM进行情绪分析：分析图片或文本的情绪，结果更加准确
- 文本生成：可以直接使用模型生成文本内容
- 多模态支持：支持文本、图片、音频、视频输入，支持文本和音频输出
- 多种模式：支持API调用模式、本地模型模式和规则模式（不使用LLM）
- 动态配置：支持在运行时更改配置参数

## 使用方法

### 1. 启用LLM功能

首先需要安装必要的依赖。编辑`requirements.txt`文件，取消LLM相关依赖的注释，然后运行：

```bash
pip install -r requirements.txt
```

### 2. 配置模式

有三种方式配置LLM：

#### 方式一：使用.env文件（推荐）

项目支持使用`.env`文件进行配置，这是最方便的方式：

1. 在`backend`目录下复制`.env.example`文件为`.env`
   ```bash
   cp .env.example .env
   ```

2. 编辑`.env`文件，根据需要修改配置项：
   ```
   # 选择模式: "api"、"local"或"rule"
   LLM_MODE=api
   
   # 如使用API模式，填写API密钥
   LLM_API_KEY=your_api_key_here
   
   # 如使用本地模式，填写模型路径
   LLM_MODEL_PATH=/path/to/model
   ```

3. 保存文件后重启服务，配置会自动加载

#### 方式二：使用环境变量

可以通过环境变量设置配置：

```bash
# Windows
set LLM_MODE=api
set LLM_API_KEY=your_api_key_here

# Linux/Mac
export LLM_MODE=api
export LLM_API_KEY=your_api_key_here
```

#### 方式三：通过API更新配置

也可以通过API动态更新配置：

```bash
curl -X POST http://localhost:5000/api/llm/config \
  -H "Content-Type: application/json" \
  -d '{"mode": "api", "api": {"api_key": "your_api_key_here"}}'
```

### 3. 可配置项说明

以下是主要的配置项及其说明：

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `LLM_MODE` | 使用模式：api、local或rule | rule |
| `LLM_API_KEY` | API密钥 | 空 |
| `LLM_API_URL` | API基础URL | https://dashscope.aliyuncs.com/... |
| `LLM_MODEL_PATH` | 本地模型路径 | 空 |
| `LLM_DEVICE` | 设备类型：auto、cuda或cpu | auto |
| `LLM_USE_QUANTIZATION` | 是否启用量化：1启用，0禁用 | 1 |
| `LLM_MAX_TOKENS` | 最大生成token数 | 1024 |
| `LLM_TEMPERATURE` | 采样温度 | 0.7 |
| `LLM_TOP_P` | 核采样参数 | 0.9 |

完整的配置项可在`.env.example`文件中查看。

## API接口

### 1. 生成文本

```
POST /api/generate-text
```

请求体：
```json
{
  "prompt": "请介绍一下通义千问2.5-Omni模型",
  "system_prompt": "你是一个专业的AI助手",
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### 2. 多模态生成

```
POST /api/generate-multimodal
```

使用表单数据（multipart/form-data）上传，支持文本、图片、音频和视频输入，可选音频输出。

请求参数：
- `prompt`：用户提示文本（必需）
- `system_prompt`：系统提示（可选）
- `max_tokens`：最大生成token数（可选）
- `temperature`：温度参数（可选）
- `top_p`：核采样参数（可选）
- `with_audio`：是否生成音频，取值`true`或`false`（可选，默认`false`）
- `voice`：音频声音，可选项为`Cherry`、`Lyra`、`Alto`、`Tenor`、`Baritone`、`Clint`（可选，默认`Cherry`）
- `image_file`：图片文件（可选，支持png、jpg、jpeg、webp）
- `audio_file`：音频文件（可选，支持mp3、wav、ogg、m4a）
- `video_file`：视频文件（可选，支持mp4、webm、avi、mov）

响应示例：
```json
{
  "text": "生成的文本内容",
  "audio": "base64编码的音频数据(如果请求了音频)",
  "model": "使用的模型名称",
  "processing_time": "处理时间"
}
```

#### 多模态JSON请求示例

如果您使用OpenAI库或直接通过JSON请求，可以参考以下格式：

##### 文本输入
```json
{
  "model": "qwen-omni-turbo",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "你是一个专业的AI助手"}]
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "请讲一个简短的故事"}]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "modalities": ["text"],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

##### 图片输入
```json
{
  "model": "qwen-omni-turbo",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "你是一个专业的AI助手"}]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片中的内容"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJ..."}}
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "modalities": ["text"],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

##### 音频输入
```json
{
  "model": "qwen-omni-turbo",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "以下是一段音频，请分析其内容并回答问题。"}],
      "content": [
        {"type": "text", "text": "以下是一段音频，请分析其内容并回答问题。"},
        {"type": "input_audio", "input_audio": {"url": "data:audio/wav;base64,UklGRiT..."}}
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "modalities": ["text"],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

##### 视频输入
```json
{
  "model": "qwen-omni-turbo",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "你是一个专业的AI助手"}]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这个视频中发生了什么"},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXBpc29t..."}}
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "modalities": ["text"],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

##### 多帧图像输入（模拟视频）
```json
{
  "model": "qwen-omni-turbo",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "你是一个专业的AI助手"}]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这个视频中的动作过程"},
        {
          "type": "video",
          "video": [
            "data:image/jpeg;base64,/9j/4AAQSkZJ...",
            "data:image/jpeg;base64,/9j/4AAQSkZJ...",
            "data:image/jpeg;base64,/9j/4AAQSkZJ..."
          ]
        }
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "modalities": ["text"],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

##### 文本输入并生成音频输出
```json
{
  "model": "qwen-omni-turbo",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "你是一个专业的AI助手"}]
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "请讲一个简短的故事"}]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "modalities": ["text", "audio"],
  "audio": {"voice": "Cherry", "format": "wav"},
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

### 3. 多帧图像处理（模拟视频）

```
POST /api/multimodal/image-frames
```

使用表单数据（multipart/form-data）上传多张图片作为视频帧序列。

请求参数：
- `prompt`：用户提示文本（必需）
- `system_prompt`：系统提示（可选）
- `max_tokens`：最大生成token数（可选）
- `temperature`：温度参数（可选）
- `top_p`：核采样参数（可选）
- `with_audio`：是否生成音频，取值`true`或`false`（可选，默认`false`）
- `voice`：音频声音（可选，默认`Cherry`）
- `image_files`：多个图片文件（必需，支持png、jpg、jpeg、webp）

响应格式与多模态生成相同。

### 4. 获取音频输出

```
GET /api/multimodal/audio-output?audio_base64=<base64编码的音频数据>
```

将base64编码的音频数据转换为音频文件并下载。

参数：
- `audio_base64`：由多模态生成接口返回的音频base64数据

响应：
- Content-Type: audio/wav
- 音频文件内容

### 5. 配置管理

获取当前配置：
```
GET /api/llm/config
```

更新配置：
```
POST /api/llm/config
```

请求体示例：
```json
{
  "mode": "api",
  "api": {
    "api_key": "你的API密钥"
  }
}
```

### 6. 情绪分析接口

图片情绪分析：
```
POST /api/analyze-emotion
```
(使用multipart/form-data表单上传图片文件)

文本情绪分析：
```
POST /api/analyze-text
```

请求体：
```json
{
  "text": "要分析的文本内容"
}
```

## 多模态使用示例

### 文本输入并生成音频输出

```bash
curl -X POST http://localhost:5000/api/generate-multimodal \
  -F "prompt=请讲一个简短的故事" \
  -F "with_audio=true" \
  -F "voice=Cherry"
```

### 图片输入

```bash
curl -X POST http://localhost:5000/api/generate-multimodal \
  -F "prompt=描述这张图片中的内容" \
  -F "image_file=@/path/to/image.jpg"
```

### 视频输入

```bash
curl -X POST http://localhost:5000/api/generate-multimodal \
  -F "prompt=描述这个视频中发生了什么" \
  -F "video_file=@/path/to/video.mp4"
```

### 多帧图像（视频）输入

```bash
curl -X POST http://localhost:5000/api/multimodal/image-frames \
  -F "prompt=描述这个视频中的动作过程" \
  -F "image_files=@/path/to/frame1.jpg" \
  -F "image_files=@/path/to/frame2.jpg" \
  -F "image_files=@/path/to/frame3.jpg"
```

## 故障排除

1. **依赖安装问题**：
   - 确保Python版本≥3.8
   - 对于Windows，可能需要从预编译轮子安装PyTorch
   
2. **内存不足**：
   - 对于本地模式，运行7B参数的模型至少需要16GB内存
   - 可通过设置`LLM_USE_QUANTIZATION=1`降低内存需求
   
3. **API调用失败**：
   - 检查API密钥是否正确
   - 确认网络连接正常
   - 查看服务日志获取详细错误信息
   
4. **.env文件未生效**：
   - 确保已安装`python-dotenv`库
   - 检查文件路径和权限
   - 尝试使用绝对路径指定.env文件位置 