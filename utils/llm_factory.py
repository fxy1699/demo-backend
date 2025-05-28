import os
import logging
from typing import Optional, Dict, Any
import sys
import pathlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_factory")

# Add the current directory to the path to ensure absolute imports work
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# 导入配置
try:
    # Try absolute import first
    from config.llm_config import get_llm_config
except ImportError:
    try:
        # Then try relative import as fallback
        from ..config.llm_config import get_llm_config
    except ImportError:
        logger.error("无法导入llm_config，请检查文件路径")
        # Define a fallback get_llm_config function
        def get_llm_config():
            return {"mode": "rule"}

# 在其他导入语句之后
from .llm_handler import LLMHandler  # 假设 LLMHandler 定义在同目录的 llm_handler.py 中
# 或者
# from ..handlers.llm_handler import LLMHandler # 假设定义在 backend/handlers/llm_handler.py 中

# 延迟导入LLMHandler，以避免在不需要时导入依赖
def get_llm_handler(config_override: Optional[Dict[str, Any]] = None) -> Any:
    """
    获取LLM处理器实例
    
    支持的模式:
        - api: 通过API调用云端大模型
        - local: 使用本地部署的Hugging Face模型
        - ollama: 使用本地Ollama实例
        - rule: 使用基于规则的处理器
    
    参数:
        config_override: 覆盖默认配置的参数（可选）
        
    返回:
        LLM处理器实例
    """
    # 获取配置
    config = get_llm_config()
    
    # 如果提供了覆盖配置，则合并配置
    if config_override is not None:
        for key, value in config_override.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    
    # 获取工作模式
    mode = config.get("mode", "rule")
    
    # 规则模式
    if mode == "rule":
        logger.info("使用规则模式")
        return get_rule_based_handler()
    
    # API模式
    if mode == "api":
        logger.info("使用API模式调用大模型")
        api_config = config.get("api", {})
        
        # 检查API密钥
        api_key = api_config.get("api_key")
        if not api_key:
            logger.warning("API模式需要提供api_key，但未找到有效的API密钥，将回退到规则模式")
            return get_rule_based_handler()
            
        # 检查API基础URL
        api_base_url = api_config.get("api_base_url")
        if not api_base_url:
            logger.warning("API模式需要提供api_base_url，但未找到有效的API基础URL，将回退到规则模式")
            return get_rule_based_handler()
        
        try:
            # 创建LLMHandler实例
            handler = LLMHandler(
                mode="api",
                api_key=api_key,
                api_base_url=api_base_url,
                model_name=api_config.get("model_name", "qwen-omni-turbo")
            )
            
            return handler
        except Exception as e:
            logger.error(f"创建API模式处理器失败: {str(e)}，将回退到规则模式")
            return get_rule_based_handler()
    
    # Ollama模式
    elif mode == "ollama":
        # 检查ollama是否可用
        try:
            from utils.ollama_handler import OllamaHandler
            OLLAMA_AVAILABLE = True
        except ImportError:
            OLLAMA_AVAILABLE = False
            logger.warning("未找到OllamaHandler，将无法使用Ollama模式")
        
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama功能未启用，回退到规则模式")
            return get_rule_based_handler()
        
        logger.info("使用Ollama模式调用本地模型")
        ollama_config = config.get("ollama", {})
        
        # 创建OllamaHandler实例
        handler = OllamaHandler(
            base_url=ollama_config.get("base_url", "http://127.0.0.1:11434"),
            api_key=ollama_config.get("api_key")
        )
        
        return handler
    
    # 本地模式
    elif mode == "local":
        logger.info("使用本地模式调用大模型")
        local_config = config.get("local", {})
        
        # 创建LLMHandler实例
        handler = LLMHandler(
            mode="local",
            local_model_path=local_config.get("model_path"),
            device=local_config.get("device", "auto"),
            use_quantization=local_config.get("use_quantization", True)
        )
        
        return handler
    
    # 未知模式
    else:
        logger.warning(f"未知模式 {mode}，回退到规则模式")
        return get_rule_based_handler()

def get_rule_based_handler():
    """获取基于规则的处理器"""
    try:
        # 导入原始的处理器
        from .emotion_generator import generate_emotion
        from .text_analyzer import analyze_text
        
        # 创建一个与LLMHandler接口兼容的包装器
        class RuleBasedHandler:
            """使用原有规则的处理器包装类"""
            
            def __init__(self):
                self.mode = "rule"
            
            def analyze_emotion_with_llm(self, image_info, system_prompt=None):
                """包装generate_emotion函数"""
                # 从image_info提取图片路径（如果有）
                image_path = image_info.get("image_path", None)
                
                if not image_path:
                    return {
                        'emotion': 'unknown',
                        'sentiment_score': 0.0,
                        'keywords': ['未知'],
                        'summary': "图片路径无效",
                        'processing_time': "0.00秒",
                        'error': "Invalid image path"
                    }
                    
                # 调用原始函数
                result = generate_emotion(image_path)
                
                return result
            
            def analyze_text_with_llm(self, text, system_prompt=None):
                """包装analyze_text函数"""
                # 调用原始函数
                result = analyze_text(text)
                
                return result
                
            def generate_text(self, prompt, system_prompt=None, max_tokens=None, 
                             temperature=None, top_p=None):
                """模拟生成文本"""
                return {
                    "text": "规则模式不支持生成文本",
                    "model": "rule-based",
                    "source": "rule"
                }
            
            def generate_multimodal(self, content, system_prompt=None, max_tokens=None,
                                  temperature=None, top_p=None, modalities=None, audio_options=None):
                """模拟多模态生成"""
                return {
                    "text": "规则模式不支持多模态生成",
                    "model": "rule-based",
                    "source": "rule"
                }
            
            def generate_text_with_image(self, prompt, image_path, system_prompt=None, 
                                       max_tokens=None, temperature=None, top_p=None,
                                       with_audio=False, voice="Cherry"):
                """模拟图像+文本生成"""
                return {
                    "text": "规则模式不支持图像+文本生成",
                    "model": "rule-based",
                    "source": "rule"
                }
            
            def generate_text_with_audio(self, prompt, audio_path, system_prompt=None, 
                                      max_tokens=None, temperature=None, top_p=None,
                                      with_audio=False, voice="Cherry"):
                """模拟音频+文本生成"""
                return {
                    "text": "规则模式不支持音频+文本生成",
                    "model": "rule-based",
                    "source": "rule"
                }
            
            def generate_text_with_video(self, prompt, video_path, system_prompt=None, 
                                      max_tokens=None, temperature=None, top_p=None,
                                      with_audio=False, voice="Cherry"):
                """模拟视频+文本生成"""
                return {
                    "text": "规则模式不支持视频+文本生成",
                    "model": "rule-based",
                    "source": "rule"
                }
            
            def generate_text_with_image_frames(self, prompt, image_paths, system_prompt=None, 
                                             max_tokens=None, temperature=None, top_p=None,
                                             with_audio=False, voice="Cherry"):
                """模拟多帧图像+文本生成"""
                return {
                    "text": "规则模式不支持多帧图像+文本生成",
                    "model": "rule-based",
                    "source": "rule"
                }
        
        return RuleBasedHandler()
        
    except ImportError as e:
        logger.error(f"导入规则处理器失败: {str(e)}")
        
        # 创建一个空壳处理器，避免程序崩溃
        class EmptyHandler:
            """空处理器"""
            
            def __init__(self):
                self.mode = "empty"
            
            def analyze_emotion_with_llm(self, image_info, system_prompt=None):
                return {
                    'emotion': 'unknown',
                    'sentiment_score': 0.0,
                    'keywords': ['未知'],
                    'summary': "处理器加载失败",
                    'processing_time': "0.00秒",
                    'error': str(e)
                }
            
            def analyze_text_with_llm(self, text, system_prompt=None):
                return {
                    'emotion': 'unknown',
                    'sentiment_score': 0.0,
                    'keywords': ['未知'],
                    'summary': "处理器加载失败",
                    'processing_time': "0.00秒",
                    'error': str(e)
                }
                
            def generate_text(self, prompt, system_prompt=None, max_tokens=None, 
                             temperature=None, top_p=None):
                return {
                    "text": "处理器加载失败",
                    "error": str(e),
                    "source": "empty"
                }
            
            def generate_multimodal(self, content, system_prompt=None, max_tokens=None,
                                  temperature=None, top_p=None, modalities=None, audio_options=None):
                return {
                    "text": "处理器加载失败",
                    "error": str(e),
                    "source": "empty"
                }
            
            def generate_text_with_image(self, prompt, image_path, system_prompt=None, 
                                       max_tokens=None, temperature=None, top_p=None,
                                       with_audio=False, voice="Cherry"):
                return {
                    "text": "处理器加载失败",
                    "error": str(e),
                    "source": "empty"
                }
            
            def generate_text_with_audio(self, prompt, audio_path, system_prompt=None, 
                                      max_tokens=None, temperature=None, top_p=None,
                                      with_audio=False, voice="Cherry"):
                return {
                    "text": "处理器加载失败",
                    "error": str(e),
                    "source": "empty"
                }
            
            def generate_text_with_video(self, prompt, video_path, system_prompt=None, 
                                      max_tokens=None, temperature=None, top_p=None,
                                      with_audio=False, voice="Cherry"):
                return {
                    "text": "处理器加载失败",
                    "error": str(e),
                    "source": "empty"
                }
            
            def generate_text_with_image_frames(self, prompt, image_paths, system_prompt=None, 
                                             max_tokens=None, temperature=None, top_p=None,
                                             with_audio=False, voice="Cherry"):
                return {
                    "text": "处理器加载失败",
                    "error": str(e),
                    "source": "empty"
                }
        
        return EmptyHandler() 