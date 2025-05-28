import os
import time
import json
import requests
import base64
from typing import Dict, List, Union, Optional, Any, Tuple
import logging
import io
import tempfile
import shutil
import subprocess
import platform
from PIL import Image

# Import prompt manager for centralized prompts
from utils.prompt_manager import get_system_prompt, get_emotion_prompt

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_handler")

# 尝试导入可选依赖
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
    import torch
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers库已加载，支持本地模型")
except ImportError:
    logger.warning("Transformers库未安装，只能使用API模式。如需本地模型支持，请安装: pip install transformers torch Pillow numpy")
    TRANSFORMERS_AVAILABLE = False

# 将视频处理工具设置为可选加载
try:
    import cv2
    VIDEO_PROCESSOR_AVAILABLE = True
    logger.info("视频处理库已加载")
except ImportError:
    logger.warning("未找到OpenCV库，视频处理功能受限。如需视频处理支持，请安装: pip install opencv-python")
    VIDEO_PROCESSOR_AVAILABLE = False

# 尝试导入OpenAI客户端
try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI客户端未安装，将使用requests库。如需使用OpenAI客户端，请安装: pip install openai")
    OPENAI_CLIENT_AVAILABLE = False

class LLMHandler:
    """大语言模型处理器，支持API调用和本地模型调用，支持多模态输入输出"""
    
    def __init__(self, 
                 mode: str = "api", 
                 api_key: Optional[str] = None,
                 api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model_name: str = "qwen-omni-turbo",
                 local_model_path: Optional[str] = None,
                 device: str = "auto",
                 use_quantization: bool = True):
        """
        初始化LLM处理器
        
        参数:
            mode: 调用模式，"api" 或 "local"
            api_key: API密钥（当mode="api"时必须提供）
            api_base_url: API基础URL
            model_name: 模型名称，默认使用商业版"qwen-omni-turbo"，也可使用开源版"qwen2.5-omni-7b"
            local_model_path: 本地模型路径（当mode="local"时必须提供）
            device: 设备类型，"auto"、"cuda"、"cpu"
            use_quantization: 是否使用量化加速（仅在local模式下有效）
        """
        self.mode = mode
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.device = device
        self.use_quantization = use_quantization
        
        # 本地模型和处理器
        self.model = None
        self.processor = None
        
        # 语音合成声音选项
        self.voice_options = ["Cherry", "Lyra", "Alto", "Tenor", "Baritone", "Clint"]
        
        # 创建一个模拟的客户端对象，防止客户端为None时出错
        class MockCompletions:
            def create(self, **kwargs):
                logger.error("使用了模拟客户端，这表明真实客户端未初始化")
                raise ValueError("OpenAI客户端未初始化，无法生成内容")
                
        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()
                
        class MockOpenAIClient:
            def __init__(self):
                self.chat = MockChat()
        
        # 初始化OpenAI客户端
        self.client = MockOpenAIClient()  # 默认使用模拟客户端
        
        if self.mode == "api" and OPENAI_CLIENT_AVAILABLE:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url
                )
                logger.info("已初始化OpenAI客户端")
            except Exception as e:
                logger.error(f"初始化OpenAI客户端失败: {str(e)}，将回退到requests库")
        
        # 初始化模型
        if self.mode == "local":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("本地模式需要安装transformers和torch库")
            
            if not self.local_model_path:
                raise ValueError("本地模式需要提供local_model_path")
                
            self._load_local_model()
        elif self.mode == "api":
            if not self.api_key:
                raise ValueError("API模式需要提供api_key")
        else:
            raise ValueError(f"不支持的模式: {mode}，请使用'api'或'local'")
    
    def _load_local_model(self):
        """加载本地模型"""
        logger.info(f"正在加载本地模型: {self.local_model_path}")
        
        try:
            # 检查设备
            if self.device == "auto":
                device_map = "auto"
            elif self.device == "cuda":
                if torch.cuda.is_available():
                    device_map = {"": 0}  # 使用第一个GPU
                else:
                    logger.warning("CUDA不可用，回退到CPU")
                    device_map = {"": "cpu"}
            else:  # cpu
                device_map = {"": "cpu"}
            
            # 量化配置
            quantization_config = None
            if self.use_quantization and device_map != {"": "cpu"}:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
            # 加载处理器
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.local_model_path)
            
            # 加载模型
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.local_model_path,
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            logger.info("本地模型加载完成")
        except Exception as e:
            logger.error(f"加载本地模型时出错: {str(e)}")
            raise
    
    def generate_text(self, 
                      prompt: str, 
                      system_prompt: str = None, 
                      max_tokens: int = 1024,
                      temperature: float = 0.7,
                      top_p: float = 0.9) -> Dict[str, Any]:
        """
        生成文本响应
        
        参数:
            prompt: 用户提示
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            
        返回:
            包含生成文本和相关信息的字典
        """
        # Use default system prompt from prompt_manager if none provided
        if system_prompt is None:
            system_prompt = get_system_prompt('llm_default')
        
        # 确保prompt是字符串
        if not isinstance(prompt, str):
            prompt = str(prompt)
        
        # 准备内容
        content = [{"type": "text", "text": prompt}]
        
        # 调用多模态生成方法
        return self.generate_multimodal(
            content=content,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
    
    def generate_multimodal(self,
                           content: List[Dict[str, Any]],
                           system_prompt: str = None,
                           max_tokens: int = 1024,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           modalities: List[str] = ["text"],
                           audio_options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        生成多模态响应
        
        参数:
            content: 输入内容列表，每个元素是一个字典，包含type和对应内容
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            modalities: 输出模态列表
            audio_options: 音频选项，仅当modalities包含audio时有效
            
        返回:
            包含生成内容和相关信息的字典
        """
        # Use default system prompt from prompt_manager if none provided
        if system_prompt is None:
            system_prompt = get_system_prompt('multimodal')
        
        start_time = time.time()
        
        # 根据模式选择不同的处理方法
        if self.mode == "api":
            # 大部分API实现都通过dashscope/openai格式，尝试先用它
            try:
                result = self._generate_multimodal_api_openai(content, system_prompt, max_tokens, temperature, top_p, modalities, audio_options)
            except Exception as e:
                logger.warning(f"使用OpenAI格式API调用失败: {str(e)}，尝试使用请求库")
                result = self._generate_multimodal_api_requests(content, system_prompt, max_tokens, temperature, top_p, modalities, audio_options)
        elif self.mode == "local":
            result = self._generate_multimodal_local(content, system_prompt, max_tokens, temperature, top_p, modalities, audio_options)
        else:
            # 不支持的模式
            return {
                "text": f"模式 {self.mode} 不支持多模态生成",
                "error": f"Unsupported mode: {self.mode}",
                "model": self.model_name if self.model_name else "unknown"
            }
            
        # 计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 添加处理时间到结果
        result["processing_time"] = f"{processing_time:.2f}秒"
        
        return result
    
    def _generate_multimodal_api_openai(self, content, system_prompt, max_tokens, temperature, top_p, modalities, audio_options):
        """使用OpenAI客户端通过API生成多模态内容"""
        logger.info("使用OpenAI客户端通过API生成多模态内容")
        
        try:
            # 确保消息格式正确
            # 1. 系统提示必须是字符串
            if not isinstance(system_prompt, str):
                system_prompt = str(system_prompt)
            
            # 2. 系统消息格式 - 直接使用字符串，不需要包装成列表和字典
            # 系统消息应该是纯文本字符串，而不是列表
            
            # 3. 确保用户内容中的text字段都是字符串，并调整image_url的格式
            user_content = []
            for item in content:
                new_item = item.copy()  # 复制一份以避免修改原始数据
                
                # 确保文本是字符串
                if new_item.get("type") == "text" and not isinstance(new_item.get("text"), str):
                    new_item["text"] = str(new_item["text"])
                
                # 检查图像URL格式
                if new_item.get("type") == "image_url" and "image_url" in new_item:
                    image_url_obj = new_item["image_url"]
                    
                    # 确保使用的是"url"而不是"urls"
                    if "urls" in image_url_obj and "url" not in image_url_obj:
                        image_url_obj["url"] = image_url_obj.pop("urls")
                    
                    url = image_url_obj.get("url", "")
                    # 如果url是base64编码字符串但不是完整的data URI
                    if not url.startswith("data:") and not url.startswith("http"):
                        # 推断MIME类型
                        mime_type = "image/jpeg"  # 默认MIME类型
                        if "image_path" in item:
                            extension = os.path.splitext(item["image_path"])[1].lower()
                            if extension == ".png":
                                mime_type = "image/png"
                            elif extension == ".gif":
                                mime_type = "image/gif"
                            elif extension == ".webp":
                                mime_type = "image/webp"
                        # 构建完整的data URI
                        image_url_obj["url"] = f"data:{mime_type};base64,{url}"
                
                # 检查音频URL格式
                if new_item.get("type") == "input_audio" and "input_audio" in new_item:
                    audio_obj = new_item["input_audio"]
                    
                    # 处理不同的音频格式字段
                    # 官方示例使用的是"data"和"format"字段
                    if "urls" in audio_obj:
                        # 旧格式转换为新格式
                        data = audio_obj.pop("urls")
                        audio_obj["data"] = data
                        audio_obj["format"] = "mp3"  # 默认格式
                        logger.warning("将音频URLs字段转换为data字段以符合API要求")
                    elif "url" in audio_obj:
                        # 旧格式转换为新格式
                        data = audio_obj.pop("url")
                        audio_obj["data"] = data
                        audio_obj["format"] = "mp3"  # 默认格式
                        logger.warning("将音频URL字段转换为data字段以符合API要求")
                    
                    # 确保包含format字段
                    if "format" not in audio_obj and "data" in audio_obj:
                        # 尝试从data URI中推断格式
                        data = audio_obj["data"]
                        if data.startswith("data:audio/"):
                            # 从data URI中提取MIME类型
                            mime_type = data.split(";")[0].split("/")[-1]
                            audio_obj["format"] = mime_type
                        else:
                            # 无法推断，使用默认格式
                            audio_obj["format"] = "mp3"
                
                # 检查视频URL格式
                if new_item.get("type") == "video_url" and "video_url" in new_item:
                    video_url_obj = new_item["video_url"]
                    
                    # 确保使用的是"url"而不是"urls"
                    if "urls" in video_url_obj and "url" not in video_url_obj:
                        video_url_obj["url"] = video_url_obj.pop("urls")
                
                user_content.append(new_item)
            
            # 记录格式化后的内容以便调试
            logger.info(f"格式化后的用户内容: {json.dumps(user_content, ensure_ascii=False)}")
            
            # 4. 准备消息格式 - 修改为OpenAI API要求的格式
            # 系统消息应该使用字符串而不是列表
            messages = [
                {"role": "system", "content": system_prompt},  # 直接使用字符串
                {"role": "user", "content": user_content}     # 用户消息保持内容列表格式
            ]
            
            # 记录请求格式
            request_json = json.dumps({
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "modalities": modalities,
                "stream": True
            }, ensure_ascii=False)
            logger.info(f"API请求格式: {request_json}")
            
            # 准备音频选项
            audio_params = audio_options if audio_options else {"voice": "Cherry", "format": "wav"}
            
            # 创建请求参数
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "modalities": modalities,
                "stream": True,  # 添加流式参数
                "stream_options": {"include_usage": True}  # 添加stream_options
            }
            
            # 如果需要音频输出，添加音频选项
            if "audio" in modalities:
                params["audio"] = audio_params
            
            # 发送请求
            logger.info(f"即将发送的完整参数: {json.dumps(params, ensure_ascii=False)}")
            
            # 检查API密钥是否设置
            if not self.api_key or self.api_key == "":
                logger.error("API密钥为空，无法进行API调用")
                return {
                    "text": "发生错误: API密钥未配置",
                    "error": "API key is empty or not set",
                    "source": "api"
                }
                
            # 检查API地址是否设置
            if not self.api_base_url:
                logger.error("API基础URL为空，无法进行API调用")
                return {
                    "text": "发生错误: API基础URL未配置",
                    "error": "API base URL is empty or not set",
                    "source": "api"
                }
                
            try:
                # 尝试使用客户端进行API调用
                completion = self.client.chat.completions.create(**params)
            except ValueError as e:
                if "OpenAI客户端未初始化" in str(e):
                    logger.error("OpenAI客户端未初始化，尝试使用requests库")
                    # 尝试使用备用方法
                    return self._generate_multimodal_api_requests(content, system_prompt, max_tokens, temperature, top_p, modalities, audio_options)
                else:
                    raise
            
            # 处理响应
            combined_text = ""
            audio_base64 = None
            usage_info = None
            
            # 流式响应处理
            for chunk in completion:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    # 处理delta中的content字段
                    if hasattr(delta, "content") and delta.content is not None:
                        if isinstance(delta.content, list):
                            for item in delta.content:
                                if item.get("type") == "text":
                                    combined_text += item.get("text", "")
                                elif item.get("type") == "audio":
                                    audio_base64 = item.get("audio", "")
                        elif isinstance(delta.content, str):
                            combined_text += delta.content
                elif hasattr(chunk, "usage"):
                    usage_info = chunk.usage
            
            # 构建结果
            result = {
                "text": combined_text,
                "model": f"{self.model_name} (API)",
                "source": "api",
            }
            
            # 添加音频信息
            if audio_base64:
                result["audio"] = audio_base64
            
            # 添加使用信息
            if usage_info:
                result["input_tokens"] = getattr(usage_info, "prompt_tokens", 0) if hasattr(usage_info, "prompt_tokens") else usage_info.get("prompt_tokens", 0)
                result["output_tokens"] = getattr(usage_info, "completion_tokens", 0) if hasattr(usage_info, "completion_tokens") else usage_info.get("completion_tokens", 0)
                result["total_tokens"] = getattr(usage_info, "total_tokens", 0) if hasattr(usage_info, "total_tokens") else usage_info.get("total_tokens", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI客户端API调用出错: {str(e)}")
            logger.exception(e)  # 记录完整堆栈跟踪
            return {
                "text": f"发生错误: {str(e)}",
                "error": str(e),
                "source": "api"
            }
    
    def _generate_multimodal_api_requests(self, content, system_prompt, max_tokens, temperature, top_p, modalities, audio_options):
        """使用requests库通过API生成多模态内容"""
        logger.info("使用requests库通过API生成多模态内容")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # 确保消息格式正确
        # 1. 系统提示必须是字符串
        if not isinstance(system_prompt, str):
            system_prompt = str(system_prompt)
            
        # 2. 创建系统消息内容
        system_content = [{"type": "text", "text": system_prompt}]
        
        # 3. 确保用户内容中的text字段都是字符串
        user_content = []
        for item in content:
            new_item = item.copy()  # 复制一份以避免修改原始数据
            if new_item.get("type") == "text" and not isinstance(new_item.get("text"), str):
                new_item["text"] = str(new_item["text"])
            user_content.append(new_item)
        
        # 4. 准备消息格式
        system_message = {
            "role": "system",
            "content": system_content
        }
        
        # 设置用户消息
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        # 构建完整消息
        messages = [system_message, user_message]
        
        # 记录请求格式
        request_json = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "modalities": modalities,
            "stream": True  # 添加流式参数
        }, ensure_ascii=False)
        logger.info(f"API请求格式: {request_json}")
        
        # 构建请求体
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "modalities": modalities,
            "stream": True  # 添加流式参数
        }
        
        # 如果需要音频输出，添加音频选项
        if "audio" in modalities and audio_options is not None:
            payload["audio"] = audio_options
        elif "audio" in modalities:
            # 默认音频选项
            payload["audio"] = {"voice": "Cherry", "format": "wav"}
        
        try:
            # 完整的API URL
            api_url = f"{self.api_base_url}/chat/completions"
            response = requests.post(api_url, headers=headers, json=payload, stream=True)
            response.raise_for_status()
            
            # 处理流式响应
            combined_text = ""
            audio_base64 = None
            usage_info = None
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # 解析JSON
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    line_text = line_text[6:]  # 移除 'data: ' 前缀
                    
                try:
                    chunk = json.loads(line_text)
                    
                    # 检查是否有选择数据
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        
                        # 检查是否有文本内容
                        if 'content' in delta:
                            if isinstance(delta['content'], list):
                                for item in delta['content']:
                                    if item.get('type') == 'text':
                                        combined_text += item.get('text', '')
                                    elif item.get('type') == 'audio':
                                        audio_base64 = item.get('audio', '')
                            elif isinstance(delta['content'], str):
                                combined_text += delta['content']
                    
                    # 检查是否有使用信息
                    if 'usage' in chunk:
                        usage_info = chunk['usage']
                        
                except json.JSONDecodeError:
                    logger.warning(f"无法解析JSON响应: {line_text}")
            
            # 构建结果
            result = {
                "text": combined_text,
                "model": f"{self.model_name} (API)",
                "source": "api",
            }
            
            # 添加音频信息
            if audio_base64:
                result["audio"] = audio_base64
            
            # 添加使用信息
            if usage_info:
                result["input_tokens"] = usage_info.get("prompt_tokens", 0)
                result["output_tokens"] = usage_info.get("completion_tokens", 0)
                result["total_tokens"] = usage_info.get("total_tokens", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"API调用出错: {str(e)}")
            return {
                "text": f"发生错误: {str(e)}",
                "error": str(e),
                "source": "api"
            }
    
    def _generate_multimodal_local(self, content, system_prompt, max_tokens, temperature, top_p, modalities, audio_options):
        """通过本地模型生成多模态内容"""
        logger.info("通过本地模型生成多模态内容")
        
        if not self.model or not self.processor:
            return {
                "text": "本地模型未加载",
                "error": "model_not_loaded",
                "source": "local"
            }
        
        try:
            # 准备输入
            system_message = {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }
            
            user_message = {
                "role": "user",
                "content": content
            }
            
            messages = [system_message, user_message]
            
            # 处理输入
            model_inputs = self.processor.chat(messages=messages, return_tensors="pt")
            
            # 将输入移动到正确的设备
            if self.device == "cuda" and torch.cuda.is_available():
                model_inputs = model_inputs.to("cuda")
            
            # 生成标志
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p
            }
            
            # 检查是否需要生成音频
            if "audio" in modalities:
                voice = "Cherry"
                if audio_options and "voice" in audio_options:
                    voice = audio_options["voice"]
                generation_kwargs["generate_speech"] = True
                generation_kwargs["voice"] = voice
            
            # 生成回复
            with torch.no_grad():
                generation_output = self.model.generate(
                    **model_inputs,
                    **generation_kwargs
                )
            
            # 解码输出
            processed_outputs = self.processor.post_process_generation(
                model_outputs=generation_output,
                input_ids=model_inputs["input_ids"],
                skip_special_tokens=True
            )
            
            # 提取输出内容
            text_output = processed_outputs.get("text", "")
            
            # 计算token数量
            input_tokens = model_inputs["input_ids"].shape[1]
            output_tokens = generation_output.shape[1] - input_tokens
            
            # 构建结果
            result = {
                "text": text_output,
                "model": f"qwen2.5-omni-local",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "source": "local"
            }
            
            # 如果有音频输出，添加到结果中
            if "audio" in modalities and "audio" in processed_outputs:
                audio_data = processed_outputs["audio"]
                # 转为base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                result["audio"] = audio_base64
            
            return result
            
        except Exception as e:
            logger.error(f"本地模型生成出错: {str(e)}")
            return {
                "text": f"生成错误: {str(e)}",
                "error": str(e),
                "source": "local"
            }
    
    def process_image(self, image_path: str, use_url: bool = False, image_url: str = None) -> str:
        """
        处理图像并转换为base64字符串或直接返回URL
        
        参数:
            image_path: 图像文件路径（当use_url=False时使用）
            use_url: 是否使用URL而非base64编码
            image_url: 图像URL（当use_url=True时使用）
            
        返回:
            base64编码的图像字符串或图像URL
        """
        if use_url and image_url:
            # 直接返回图像URL
            return image_url
        
        try:
            # 使用PIL处理图像，强制压缩到API限制以下
            if TRANSFORMERS_AVAILABLE:  # 确保PIL可用
                from PIL import Image
                import io
                
                # 打开图像
                img = Image.open(image_path)
                
                # 检查图像模式，如果是RGBA（带透明通道），则转换为RGB
                if img.mode == 'RGBA':
                    logger.info("检测到RGBA图像，转换为RGB模式")
                    img = img.convert('RGB')
                
                # 计算原始尺寸
                original_width, original_height = img.size
                logger.info(f"原始图像尺寸: {original_width}x{original_height}")
                
                # 最大允许的字节数（考虑base64膨胀因子约为1.37）
                max_bytes = int(10 * 1024 * 1024 / 1.37)  # 约为7.5MB
                
                # 初始化压缩参数
                quality = 90
                max_dim = 1600
                current_bytes = os.path.getsize(image_path)
                
                # 确保图像是RGB格式用于JPEG保存
                def ensure_rgb(image):
                    if image.mode == 'RGBA':
                        logger.info("保存前转换RGBA图像为RGB模式")
                        return image.convert('RGB')
                    return image

                # 先尝试降低质量和尺寸
                if current_bytes > max_bytes:
                    # 先检查尺寸是否过大
                    if max(original_width, original_height) > max_dim:
                        # 计算缩放比例
                        scale = max_dim / max(original_width, original_height)
                        new_width = int(original_width * scale)
                        new_height = int(original_height * scale)
                        
                        # 调整大小
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        logger.info(f"调整后尺寸: {new_width}x{new_height}")

                # 以不同质量保存，直到小于限制
                buffer = io.BytesIO()
                img_to_save = ensure_rgb(img)
                img_to_save.save(buffer, format="JPEG", quality=quality)
                current_size = len(buffer.getvalue())
                
                # 如果仍然太大，继续降低质量
                while current_size > max_bytes and quality > 30:
                    buffer = io.BytesIO()
                    quality -= 10
                    img_to_save = ensure_rgb(img)
                    img_to_save.save(buffer, format="JPEG", quality=quality)
                    current_size = len(buffer.getvalue())
                    logger.info(f"降低质量到 {quality}，当前大小: {current_size/1024/1024:.2f}MB")
                
                # 如果质量已经很低但仍然太大，继续降低尺寸
                scale_factor = 0.9
                while current_size > max_bytes and scale_factor > 0.3:
                    buffer = io.BytesIO()
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_img = ensure_rgb(resized_img)
                    resized_img.save(buffer, format="JPEG", quality=quality)
                    current_size = len(buffer.getvalue())
                    logger.info(f"缩小到 {scale_factor:.1f}倍，当前大小: {current_size/1024/1024:.2f}MB")
                    scale_factor -= 0.1
                
                # 如果还是太大，使用更激进的压缩方法
                if current_size > max_bytes:
                    # 最终尝试，强制转换为灰度图像
                    logger.warning("尝试转换为灰度图像以减小大小")
                    buffer = io.BytesIO()
                    gray_img = img.convert('L')
                    gray_img.save(buffer, format="JPEG", quality=50)
                    current_size = len(buffer.getvalue())
                
                # 到这一步如果还是太大，只能截取了
                if current_size > max_bytes:
                    logger.warning("图像仍然过大，尝试裁剪")
                    # 裁剪中心区域
                    crop_factor = (max_bytes / current_size) ** 0.5
                    center_x, center_y = img.width // 2, img.height // 2
                    crop_width = int(img.width * crop_factor)
                    crop_height = int(img.height * crop_factor)
                    left = center_x - crop_width // 2
                    top = center_y - crop_height // 2
                    right = left + crop_width
                    bottom = top + crop_height
                    
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img = ensure_rgb(cropped_img)
                    buffer = io.BytesIO()
                    cropped_img.save(buffer, format="JPEG", quality=50)
                    current_size = len(buffer.getvalue())
                
                logger.info(f"最终图像大小: {current_size/1024/1024:.2f}MB")
                
                # 转为base64
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return encoded_string
            else:
                # 无PIL可用，只能直接读取并希望不超过大小
                with open(image_path, "rb") as image_file:
                    data = image_file.read()
                    if len(data) > 10 * 1024 * 1024:
                        raise ValueError(f"图像文件过大({len(data)/1024/1024:.2f}MB)且无法压缩，超过API限制(10MB)")
                    return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            logger.error(f"处理图像时出错: {str(e)}")
            raise
    
    def process_audio(self, audio_path: str, use_url: bool = False, audio_url: str = None) -> Dict[str, str]:
        """
        处理音频并转换为API格式
        
        参数:
            audio_path: 音频文件路径（当use_url=False时使用）
            use_url: 是否使用URL而非base64编码
            audio_url: 音频URL（当use_url=True时使用）
            
        返回:
            包含audio数据和格式的字典
        """
        if use_url and audio_url:
            # 获取URL格式的扩展名
            extension = audio_url.split('.')[-1].lower() if '.' in audio_url else 'wav'
            audio_format = extension
            if extension == 'm4a':
                audio_format = 'mp4'
            
            # 直接返回音频URL和格式
            return {
                "data": audio_url,
                "format": audio_format
            }
        
        try:
            # 获取文件扩展名
            extension = os.path.splitext(audio_path)[1].lower().replace('.', '')
            audio_format = extension
            if extension == 'm4a':
                audio_format = 'mp4'
            
            # 读取音频文件并转为base64
            with open(audio_path, "rb") as audio_file:
                encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
                
                # 返回data:uri格式的数据和格式
                return {
                    "data": f"data:audio/{audio_format};base64,{encoded_string}",
                    "format": audio_format
                }
        except Exception as e:
            logger.error(f"处理音频时出错: {str(e)}")
            raise
    
    def process_video(self, video_path: str) -> str:
        """
        处理视频并转换为base64字符串
        
        参数:
            video_path: 视频文件路径
            
        返回:
            base64编码的视频字符串
        """
        try:
            with open(video_path, "rb") as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
                # 根据文件扩展名确定MIME类型
                extension = os.path.splitext(video_path)[1].lower()
                mime_type = "video/mp4"  # 默认mime类型
                if extension == ".avi":
                    mime_type = "video/x-msvideo"
                elif extension == ".webm":
                    mime_type = "video/webm"
                elif extension == ".mov":
                    mime_type = "video/quicktime"
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            raise
    
    def process_image_frames(self, image_paths: List[str]) -> List[str]:
        """
        处理多个图像帧并转换为base64字符串列表
        
        参数:
            image_paths: 图像文件路径列表
            
        返回:
            base64编码的图像字符串列表
        """
        return [self.process_image(path) for path in image_paths]
        
    def generate_text_with_image(self, 
                                prompt: str, 
                                image_path: str = None,
                                image_url: str = None,
                                image_file_data = None,
                                system_prompt: str = None, 
                                max_tokens: int = 1024,
                                temperature: float = 0.7,
                                top_p: float = 0.9,
                                with_audio: bool = False,
                                voice: str = "Cherry") -> Dict[str, Any]:
        """
        生成带图像的文本响应
        
        参数:
            prompt: 用户提示
            image_path: 图像文件路径
            image_url: 图像URL
            image_file_data: 图像文件数据（二进制）
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            with_audio: 是否生成音频
            voice: 音频声音
            
        返回:
            包含生成内容和相关信息的字典
        """
        # Use default system prompt from prompt_manager if none provided
        if system_prompt is None:
            system_prompt = get_system_prompt('image')
        
        # 处理图像数据
        image_base64 = None
        if image_file_data is not None:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                temp_img.write(image_file_data)
                temp_img_path = temp_img.name
                
            # 处理图像
            try:
                image_base64 = self.process_image(temp_img_path)
                # 使用完成后删除临时文件
                os.unlink(temp_img_path)
            except Exception as e:
                logger.error(f"处理图像文件数据出错: {str(e)}")
                # 尝试删除临时文件
                try:
                    os.unlink(temp_img_path)
                except:
                    pass
        else:
            # 使用图像路径或URL
            image_base64 = self.process_image(image_path, image_path is None and image_url is not None, image_url)
        
        # 构建内容
        content = [
            {"type": "image_url", "image_url": {"url": image_base64}},
            {"type": "text", "text": prompt},
        ]
        
        # 设置输出模态
        modalities = ["text"]
        audio_options = None
        if with_audio:
            modalities.append("audio")
            audio_options = {"voice": voice, "format": "wav"}
        
        # 生成回复
        return self.generate_multimodal(
            content=content,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            modalities=modalities,
            audio_options=audio_options
        )
    
    def generate_text_with_audio(self, 
                               prompt: str, 
                               audio_path: str = None,
                               audio_url: str = None,
                               audio_file_data = None,
                               system_prompt: str = None, 
                               max_tokens: int = 1024,
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               with_audio: bool = False,
                               voice: str = "Cherry") -> Dict[str, Any]:
        """
        生成带音频的文本响应
        
        参数:
            prompt: 用户提示
            audio_path: 音频文件路径
            audio_url: 音频URL
            audio_file_data: 音频文件数据（二进制）
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            with_audio: 是否生成音频
            voice: 音频声音
            
        返回:
            包含生成内容和相关信息的字典
        """
        # Use default system prompt from prompt_manager if none provided
        if system_prompt is None:
            system_prompt = get_system_prompt('audio')
        
        # 处理音频数据
        audio_info = None
        if audio_file_data is not None:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_file_data)
                temp_audio_path = temp_audio.name
                
            # 处理音频
            try:
                audio_info = self.process_audio(temp_audio_path)
                # 使用完成后删除临时文件
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"处理音频文件数据出错: {str(e)}")
                # 尝试删除临时文件
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
        else:
            # 使用音频路径或URL
            audio_info = self.process_audio(audio_path, audio_path is None and audio_url is not None, audio_url)
        
        # 获取音频base64数据和格式
        audio_data = audio_info.get("data", "")
        audio_format = audio_info.get("format", "wav")
        
        # 构建内容
        content = [
            {"type": "input_audio", "input_audio": {"data": audio_data, "format": audio_format}},
            {"type": "text", "text": prompt}
        ]
        
        # 设置输出模态
        modalities = ["text"]
        audio_options = None
        if with_audio:
            modalities.append("audio")
            audio_options = {"voice": voice, "format": "wav"}
        
        # 生成回复
        return self.generate_multimodal(
            content=content,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            modalities=modalities,
            audio_options=audio_options
        )
    
    def generate_text_with_video(self, 
                               prompt: str, 
                               video_path: str,
                               system_prompt: str = None, 
                               max_tokens: int = 1024,
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               with_audio: bool = False,
                               voice: str = "Cherry") -> Dict[str, Any]:
        """
        生成带视频的文本响应
        
        参数:
            prompt: 用户提示
            video_path: 视频文件路径
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            with_audio: 是否生成音频
            voice: 音频声音
            
        返回:
            包含生成内容和相关信息的字典
        """
        # Use default system prompt from prompt_manager if none provided
        if system_prompt is None:
            system_prompt = get_system_prompt('video')
        
        # 处理视频
        video_base64 = self.process_video(video_path)
        
        # 构建内容
        content = [
            {"type": "video", "video": video_base64},
            {"type": "text", "text": prompt}
        ]
        
        # 设置输出模态
        modalities = ["text"]
        audio_options = None
        if with_audio:
            modalities.append("audio")
            audio_options = {"voice": voice, "format": "wav"}
        
        # 生成回复
        return self.generate_multimodal(
            content=content,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            modalities=modalities,
            audio_options=audio_options
        )
    
    def generate_text_with_image_frames(self, 
                                      prompt: str, 
                                      image_paths: List[str],
                                      system_prompt: str = None, 
                                      max_tokens: int = 1024,
                                      temperature: float = 0.7,
                                      top_p: float = 0.9,
                                      with_audio: bool = False,
                                      voice: str = "Cherry") -> Dict[str, Any]:
        """
        生成带多帧图像的文本响应（模拟视频）
        
        参数:
            prompt: 用户提示
            image_paths: 图像文件路径列表
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            with_audio: 是否生成音频
            voice: 音频声音
            
        返回:
            包含生成内容和相关信息的字典
        """
        # Use default system prompt from prompt_manager if none provided
        if system_prompt is None:
            system_prompt = get_system_prompt('video')
        
        # 处理多帧图像
        image_base64_list = self.process_image_frames(image_paths)
        
        # 构建内容
        content = [
            {"type": "video", "video": image_base64_list},
            {"type": "text", "text": prompt}
        ]
        
        # 设置输出模态
        modalities = ["text"]
        audio_options = None
        if with_audio:
            modalities.append("audio")
            audio_options = {"voice": voice, "format": "wav"}
        
        # 生成回复
        return self.generate_multimodal(
            content=content,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            modalities=modalities,
            audio_options=audio_options
        )
    
    def analyze_emotion_with_llm(self, 
                                image_info: Dict[str, Any], 
                                system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        使用LLM分析图像情绪
        
        参数:
            image_info: 图像信息字典
            system_prompt: 自定义系统提示
            
        返回:
            情绪分析结果
        """
        if not system_prompt:
            system_prompt = get_emotion_prompt('image')
        
        # 构建提示
        prompt = f"请分析这张图片的情绪：\n{json.dumps(image_info, ensure_ascii=False, indent=2)}"
        
        # 生成回复
        result = self.generate_text(prompt, system_prompt, max_tokens=512)
        
        try:
            # 尝试从回复中提取JSON
            text = result["text"]
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start < 0 or json_end <= 0:
                # 找不到JSON，尝试自己构建
                return self._parse_emotion_fallback(text)
                
            json_str = text[json_start:json_end]
            emotion_result = json.loads(json_str)
            
            # 验证关键字段
            required_fields = ['emotion', 'sentiment_score', 'keywords', 'summary']
            for field in required_fields:
                if field not in emotion_result:
                    emotion_result[field] = "未指定" if field != 'sentiment_score' else 0.0
                    
            # 确保keywords是列表
            if not isinstance(emotion_result['keywords'], list):
                emotion_result['keywords'] = [emotion_result['keywords']]
                
            # 记录处理时间
            emotion_result['processing_time'] = result.get('processing_time', '未知')
            
            return emotion_result
            
        except Exception as e:
            logger.error(f"解析情绪分析结果时出错: {str(e)}")
            return {
                'emotion': 'unknown',
                'sentiment_score': 0.0,
                'keywords': ['未知'],
                'summary': f"分析失败: {str(e)}",
                'processing_time': result.get('processing_time', '未知'),
                'error': str(e)
            }
    
    def analyze_text_with_llm(self, 
                             text: str, 
                             system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        使用LLM分析文本情绪
        
        参数:
            text: 要分析的文本
            system_prompt: 自定义系统提示
            
        返回:
            情绪分析结果
        """
        if not system_prompt:
            system_prompt = get_emotion_prompt('text')
        
        # 构建提示
        prompt = f"请分析这段文本的情绪：\n\"{text}\""
        
        # 生成回复
        result = self.generate_text(prompt, system_prompt, max_tokens=512)
        
        try:
            # 尝试从回复中提取JSON
            text_result = result["text"]
            json_start = text_result.find('{')
            json_end = text_result.rfind('}') + 1
            if json_start < 0 or json_end <= 0:
                # 找不到JSON，尝试自己构建
                return self._parse_emotion_fallback(text_result)
                
            json_str = text_result[json_start:json_end]
            emotion_result = json.loads(json_str)
            
            # 验证关键字段
            required_fields = ['emotion', 'sentiment_score', 'keywords', 'summary']
            for field in required_fields:
                if field not in emotion_result:
                    emotion_result[field] = "未指定" if field != 'sentiment_score' else 0.0
                    
            # 确保keywords是列表
            if not isinstance(emotion_result['keywords'], list):
                emotion_result['keywords'] = [emotion_result['keywords']]
                
            # 记录处理时间
            emotion_result['processing_time'] = result.get('processing_time', '未知')
            
            return emotion_result
            
        except Exception as e:
            logger.error(f"解析文本情绪分析结果时出错: {str(e)}")
            return {
                'emotion': 'unknown',
                'sentiment_score': 0.0,
                'keywords': ['未知'],
                'summary': f"分析失败: {str(e)}",
                'processing_time': result.get('processing_time', '未知'),
                'error': str(e)
            }
    
    def _parse_emotion_fallback(self, text):
        """当JSON解析失败时，尝试从文本中提取情绪信息"""
        # 情绪映射表 - 简化为三种猫头鹰情绪
        emotion_keywords = {
            'happy': ['开心', '高兴', '快乐', '幸福', '喜悦', '惊喜', '惊讶'],
            'angry': ['生气', '愤怒', '恼火', '不满', '讨厌', '平静', '平和', '无聊'],
            'sad': ['悲伤', '难过', '伤心', '痛苦', '忧伤', '失望']
        }
        
        # 默认情绪
        emotion = 'angry'  # 默认为生气/无聊
        
        # 尝试从文本中识别情绪
        emotion_counts = {e: 0 for e in emotion_keywords.keys()}
        for e, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_counts[e] += 1
        
        # 找出提到最多的情绪
        if any(emotion_counts.values()):
            emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # 提取可能的关键词
        words = text.replace('\n', ' ').split(' ')
        keywords = [w for w in words if len(w) > 1 and any(k in w for e in emotion_keywords.values() for k in e)]
        keywords = list(set(keywords))[:5]  # 去重并限制数量
        
        # 生成简单的摘要
        summary = f"基于文本分析，表达了{emotion}的情绪。"
        
        return {
            'emotion': emotion,
            'keywords': keywords or ['未知'],
            'summary': summary,
            'processing_time': '未知',
            'note': '结果由文本分析生成，非结构化JSON'
        }

    def generate_text_with_image_and_audio(self, 
                                    prompt: str, 
                                    image_file_data = None,
                                    audio_file_data = None,
                                    image_path: str = None,
                                    audio_path: str = None,
                                    image_url: str = None,
                                    audio_url: str = None,
                                    system_prompt: str = None, 
                                    max_tokens: int = 1024,
                                    temperature: float = 0.7,
                                    top_p: float = 0.9,
                                    with_audio: bool = False,
                                    voice: str = "Cherry") -> Dict[str, Any]:
        """
        生成基于图像和音频的文本响应
        
        参数:
            prompt: 用户提示
            image_file_data: 图像文件数据（二进制）
            audio_file_data: 音频文件数据（二进制）
            image_path: 图像文件路径，当image_file_data为None时使用
            audio_path: 音频文件路径，当audio_file_data为None时使用
            image_url: 图像URL，当image_file_data和image_path都为None时使用
            audio_url: 音频URL，当audio_file_data和audio_path都为None时使用
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            with_audio: 是否生成音频
            voice: 音频声音选项
            
        返回:
            包含生成文本和相关信息的字典
        """
        logger.info(f"生成基于图像和音频的文本响应")
        
        # 处理系统提示
        if system_prompt is None:
            system_prompt = get_system_prompt('multimodal')
            
        # 处理图像数据
        image_content = None
        if image_file_data is not None:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                temp_img.write(image_file_data)
                temp_img_path = temp_img.name
                
            # 处理图像
            try:
                image_content = self.process_image(temp_img_path)
                # 使用完成后删除临时文件
                os.unlink(temp_img_path)
            except Exception as e:
                logger.error(f"处理图像文件数据出错: {str(e)}")
                # 尝试删除临时文件
                try:
                    os.unlink(temp_img_path)
                except:
                    pass
                image_content = None
        elif image_path is not None:
            try:
                image_content = self.process_image(image_path)
            except Exception as e:
                logger.error(f"处理图像文件路径出错: {str(e)}")
                image_content = None
        elif image_url is not None:
            try:
                image_content = self.process_image(None, use_url=True, image_url=image_url)
            except Exception as e:
                logger.error(f"处理图像URL出错: {str(e)}")
                image_content = None
                
        # 处理音频数据
        audio_content = None
        if audio_file_data is not None:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_file_data)
                temp_audio_path = temp_audio.name
                
            # 处理音频
            try:
                audio_content = self.process_audio(temp_audio_path)
                # 使用完成后删除临时文件
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"处理音频文件数据出错: {str(e)}")
                # 尝试删除临时文件
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                audio_content = None
        elif audio_path is not None:
            try:
                audio_content = self.process_audio(audio_path)
            except Exception as e:
                logger.error(f"处理音频文件路径出错: {str(e)}")
                audio_content = None
        elif audio_url is not None:
            try:
                audio_content = self.process_audio(None, use_url=True, audio_url=audio_url)
            except Exception as e:
                logger.error(f"处理音频URL出错: {str(e)}")
                audio_content = None
        
        # 检查图像和音频内容是否可用
        if image_content is None and audio_content is None:
            return {
                "text": "无法处理提供的图像和音频数据",
                "error": "Image and audio processing failed"
            }
        
        # 准备多模态内容
        content = []
        
        # 添加用户文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 添加图像内容
        if image_content is not None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_content
                }
            })
        
        # 添加音频内容
        if audio_content is not None and isinstance(audio_content, dict) and "url" in audio_content:
            content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": audio_content["url"],
                    "format": audio_content.get("format", "wav")
                }
            })
        
        # 设置音频输出选项
        audio_options = None
        if with_audio:
            audio_options = {
                "voice": voice,
                "format": "wav"
            }
        
        # 设置请求模态
        modalities = ["text"]
        if with_audio:
            modalities.append("audio")
        
        # 使用generate_multimodal处理多模态请求
        try:
            result = self.generate_multimodal(
                content=content,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                modalities=modalities,
                audio_options=audio_options
            )
            
            # 在响应中添加源提示
            result["prompt"] = prompt
            
            return result
        except Exception as e:
            logger.error(f"多模态生成失败: {str(e)}")
            error_message = "无法处理图像和音频内容"
            return {
                "text": error_message,
                "error": str(e),
                "prompt": prompt
            } 