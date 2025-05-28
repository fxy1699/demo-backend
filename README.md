# 猫头鹰情绪展示系统 - 后端

这是猫头鹰情绪展示系统的后端服务，主要功能是接收前端上传的图片或文本，生成情绪分析结果，并将结果返回给前端。

## 功能特性

- 支持图片上传与处理
- 支持文本情绪分析
- 随机生成情绪分析结果
- 提供RESTful API接口
- 支持CORS跨域请求

## 技术栈

- **Flask**: 轻量级Python Web框架
- **Python 3.x**: 主要开发语言
- **Pillow**: 图像处理库
- **jieba**: 中文分词库

## 项目结构

```
backend/
├── app.py                # 应用主文件
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── emotion_generator.py  # 图像情绪生成模块
│   └── text_analyzer.py      # 文本分析模块
├── static/               # 静态资源
│   └── uploads/          # 上传文件存储
├── .gitignore            # Git忽略文件
└── requirements.txt      # 依赖列表
```

## 安装与运行

### 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 运行应用

```bash
python app.py
```

应用将在 http://localhost:5000 上运行。

## API说明

### 1. 图片上传与情绪分析接口

```
POST /api/analyze-emotion
```

**请求参数**:
- Content-Type: multipart/form-data
- 参数: file (图片文件，支持PNG、JPG格式)

**成功响应**:
```json
{
  "emotion": "happy",
  "sentiment_score": 0.75,
  "keywords": ["微笑", "愉快", "积极"],
  "summary": "图片中显示了微笑和积极的氛围，整体表现出愉快的情绪。",
  "processing_time": "1.25秒",
  "image_info": {
    "dimensions": "800x600",
    "format": "JPEG",
    "mode": "RGB",
    "size": "125.50 KB"
  }
}
```

**错误响应**:
```json
{
  "error": "错误信息"
}
```

### 2. 文本情绪分析接口

```
POST /api/analyze-text
```

**请求参数**:
- Content-Type: application/json
- 请求体:
```json
{
  "text": "需要分析的文本内容"
}
```

**成功响应**:
```json
{
  "emotion": "happy",
  "sentiment_score": 0.78,
  "keywords": ["开心", "愉快", "微笑", "喜欢", "好"],
  "summary": "文本表达了开心和愉快的情绪，整体氛围积极向上。",
  "processing_time": "0.35秒",
  "text_stats": {
    "word_count": 15,
    "char_count": 25,
    "emotion_distribution": {
      "happy": 0.6,
      "angry": 0.1,
      "sad": 0.05,
      "surprised": 0.15,
      "calm": 0.1
    }
  }
}
```

**错误响应**:
```json
{
  "error": "错误信息"
}
```

### 3. 健康检查接口

```
GET /api/healthcheck
```

**成功响应**:
```json
{
  "status": "ok",
  "message": "服务正常运行中"
}
```

## 注意事项

- 为简化实现，本服务使用随机生成的方式模拟情绪分析结果，不涉及真实的AI模型。
- 上传的图片文件将保存在`static/uploads`目录中。
- 文本分析使用jieba进行中文分词，并使用简单的基于关键词的规则进行情绪判断。
- 默认情况下，服务允许所有来源的跨域请求。
- 文本长度限制为5000字符。 