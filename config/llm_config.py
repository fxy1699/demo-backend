import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv, set_key

# 设置日志
logger = logging.getLogger("llm_config")

# 尝试加载.env文件
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"已加载配置文件: {env_path}")
else:
    logger.warning(f"未找到.env文件: {env_path}")

# Import the prompt manager for system prompts
try:
    from utils.prompt_manager import get_system_prompt, SYSTEM_PROMPTS
    PROMPT_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("未找到prompt_manager模块，使用默认提示")
    PROMPT_MANAGER_AVAILABLE = False

# 默认配置
DEFAULT_CONFIG = {
    "mode": "ollama",  # Changed default to ollama mode
    "api": {
        "api_key": "",
        "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model_name": "qwen-omni-turbo"
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "api_key": "",
        "models": {
            "text": {
                "name": "qwen2.5:7b",
                "context_window": 8192,
                "system_prompt": SYSTEM_PROMPTS['ollama_text'] if PROMPT_MANAGER_AVAILABLE else "You are a helpful AI assistant."
            },
            "multimodal": {
                "name": "llava:7b",
                "context_window": 4096,
                "system_prompt": SYSTEM_PROMPTS['ollama_multimodal'] if PROMPT_MANAGER_AVAILABLE else "You are a helpful multimodal AI assistant that can understand both images and text."
            }
        },
        "default_model": "text"  # Default to text model unless image input is detected
    },
    "local": {
        "model_path": "",
        "device": "auto",
        "use_quantization": True
    },
    "generation": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    },
    "prompts": {
        "emotion_text": None,  # Will be loaded from prompt_manager
        "emotion_image": None  # Will be loaded from prompt_manager
    }
}

# 配置缓存
_config_cache = None

def get_llm_config() -> Dict[str, Any]:
    """
    获取LLM配置
    
    优先从环境变量读取，如果环境变量不存在则使用默认值
    
    返回:
        包含配置信息的字典
    """
    global _config_cache
    
    # 如果缓存存在，直接返回
    if _config_cache is not None:
        return _config_cache
    
    # 初始化配置（复制默认配置）
    config = DEFAULT_CONFIG.copy()
    
    # 从环境变量读取配置
    
    # 基本配置
    config["mode"] = os.getenv("LLM_MODE", config["mode"])
    
    # API配置
    config["api"]["api_key"] = os.getenv("LLM_API_KEY", config["api"]["api_key"])
    config["api"]["api_base_url"] = os.getenv("LLM_API_URL", config["api"]["api_base_url"])
    config["api"]["model_name"] = os.getenv("LLM_MODEL_NAME", config["api"]["model_name"])
    
    # Ollama配置
    config["ollama"]["base_url"] = os.getenv("LLM_OLLAMA_BASE_URL", config["ollama"]["base_url"])
    config["ollama"]["api_key"] = os.getenv("LLM_OLLAMA_API_KEY", config["ollama"]["api_key"])
    config["ollama"]["models"]["text"]["name"] = os.getenv("LLM_OLLAMA_TEXT_MODEL_NAME", config["ollama"]["models"]["text"]["name"])
    config["ollama"]["models"]["multimodal"]["name"] = os.getenv("LLM_OLLAMA_MULTIMODAL_MODEL_NAME", config["ollama"]["models"]["multimodal"]["name"])
    
    # 本地模型配置
    config["local"]["model_path"] = os.getenv("LLM_MODEL_PATH", config["local"]["model_path"])
    config["local"]["device"] = os.getenv("LLM_DEVICE", config["local"]["device"])
    config["local"]["use_quantization"] = os.getenv("LLM_USE_QUANTIZATION", "1") == "1"
    
    # 生成配置
    try:
        config["generation"]["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", str(config["generation"]["max_tokens"])))
    except ValueError:
        pass
        
    try:
        config["generation"]["temperature"] = float(os.getenv("LLM_TEMPERATURE", str(config["generation"]["temperature"])))
    except ValueError:
        pass
        
    try:
        config["generation"]["top_p"] = float(os.getenv("LLM_TOP_P", str(config["generation"]["top_p"])))
    except ValueError:
        pass
    
    # 更新缓存
    _config_cache = config
    
    logger.info(f"已加载LLM配置，当前模式: {config['mode']}")
    return config

def update_llm_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新LLM配置
    
    参数:
        new_config: 新的配置参数
        
    返回:
        更新后的完整配置
    """
    global _config_cache
    
    # 获取当前配置
    current_config = get_llm_config()
    
    # 深度更新配置
    updated_config = _deep_update(current_config, new_config)
    
    # 更新环境变量
    if "mode" in new_config:
        os.environ["LLM_MODE"] = new_config["mode"]
        if os.path.exists(env_path):
            set_key(env_path, "LLM_MODE", new_config["mode"])
    
    # 更新API配置
    if "api" in new_config:
        if "api_key" in new_config["api"]:
            os.environ["LLM_API_KEY"] = new_config["api"]["api_key"]
            if os.path.exists(env_path):
                set_key(env_path, "LLM_API_KEY", new_config["api"]["api_key"])
                
        if "api_base_url" in new_config["api"]:
            os.environ["LLM_API_URL"] = new_config["api"]["api_base_url"]
            if os.path.exists(env_path):
                set_key(env_path, "LLM_API_URL", new_config["api"]["api_base_url"])
                
        if "model_name" in new_config["api"]:
            os.environ["LLM_MODEL_NAME"] = new_config["api"]["model_name"]
            if os.path.exists(env_path):
                set_key(env_path, "LLM_MODEL_NAME", new_config["api"]["model_name"])
    
    # 更新Ollama配置
    if "ollama" in new_config:
        if "base_url" in new_config["ollama"]:
            os.environ["LLM_OLLAMA_BASE_URL"] = new_config["ollama"]["base_url"]
            if os.path.exists(env_path):
                set_key(env_path, "LLM_OLLAMA_BASE_URL", new_config["ollama"]["base_url"])
                
        if "api_key" in new_config["ollama"]:
            os.environ["LLM_OLLAMA_API_KEY"] = new_config["ollama"]["api_key"]
            if os.path.exists(env_path):
                set_key(env_path, "LLM_OLLAMA_API_KEY", new_config["ollama"]["api_key"])
                
        if "models" in new_config["ollama"]:
            if "text" in new_config["ollama"]["models"]:
                os.environ["LLM_OLLAMA_TEXT_MODEL_NAME"] = new_config["ollama"]["models"]["text"]["name"]
                if os.path.exists(env_path):
                    set_key(env_path, "LLM_OLLAMA_TEXT_MODEL_NAME", new_config["ollama"]["models"]["text"]["name"])
                
            if "multimodal" in new_config["ollama"]["models"]:
                os.environ["LLM_OLLAMA_MULTIMODAL_MODEL_NAME"] = new_config["ollama"]["models"]["multimodal"]["name"]
                if os.path.exists(env_path):
                    set_key(env_path, "LLM_OLLAMA_MULTIMODAL_MODEL_NAME", new_config["ollama"]["models"]["multimodal"]["name"])
    
    # 更新本地模型配置
    if "local" in new_config:
        if "model_path" in new_config["local"]:
            os.environ["LLM_MODEL_PATH"] = new_config["local"]["model_path"]
            if os.path.exists(env_path):
                set_key(env_path, "LLM_MODEL_PATH", new_config["local"]["model_path"])
                
        if "device" in new_config["local"]:
            os.environ["LLM_DEVICE"] = new_config["local"]["device"]
            if os.path.exists(env_path):
                set_key(env_path, "LLM_DEVICE", new_config["local"]["device"])
                
        if "use_quantization" in new_config["local"]:
            os.environ["LLM_USE_QUANTIZATION"] = "1" if new_config["local"]["use_quantization"] else "0"
            if os.path.exists(env_path):
                set_key(env_path, "LLM_USE_QUANTIZATION", "1" if new_config["local"]["use_quantization"] else "0")
    
    # 更新生成配置
    if "generation" in new_config:
        if "max_tokens" in new_config["generation"]:
            os.environ["LLM_MAX_TOKENS"] = str(new_config["generation"]["max_tokens"])
            if os.path.exists(env_path):
                set_key(env_path, "LLM_MAX_TOKENS", str(new_config["generation"]["max_tokens"]))
                
        if "temperature" in new_config["generation"]:
            os.environ["LLM_TEMPERATURE"] = str(new_config["generation"]["temperature"])
            if os.path.exists(env_path):
                set_key(env_path, "LLM_TEMPERATURE", str(new_config["generation"]["temperature"]))
                
        if "top_p" in new_config["generation"]:
            os.environ["LLM_TOP_P"] = str(new_config["generation"]["top_p"])
            if os.path.exists(env_path):
                set_key(env_path, "LLM_TOP_P", str(new_config["generation"]["top_p"]))
    
    # 更新缓存
    _config_cache = updated_config
    
    logger.info(f"已更新LLM配置，当前模式: {updated_config['mode']}")
    return updated_config

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """深度更新字典"""
    result = d.copy()
    for k, v in u.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result 