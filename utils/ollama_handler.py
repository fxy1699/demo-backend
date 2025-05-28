"""
Ollama API integration module with memory features
"""

import os
import requests
import json
import logging
import base64
import time
import threading
from typing import Dict, List, Union, Optional, Any
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ollama_handler")

class OllamaHandler:
    """Ollama API handler with memory features"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", api_key: Optional[str] = None):
        """
        Initialize Ollama API handler
        
        Args:
            base_url: Ollama service base URL
            api_key: Ollama API key (if required)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {}
        self.conversation_history = []
        self.monitoring_active = False
        self.tokens_generated = 0
        self.token_timestamps = []
        self.generation_start_time = 0
        self.estimated_total_tokens = 500  # Default value
        self.gpu_enabled = False
        self.gpu_info = self._detect_gpu()
        self.chat = None  # 确保chat属性初始化
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Try to enable GPU execution
        self.ensure_gpu_execution()
        
        logger.info(f"Initialized Ollama handler with base URL: {base_url}")
        if self.gpu_info:
            logger.info(f"Detected GPU: {self.gpu_info}")
    
    def _detect_gpu(self) -> Optional[str]:
        """Detect available GPU hardware."""
        try:
            # Try NVIDIA GPU detection
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
                
        except FileNotFoundError:
            try:
                # Check for Apple Silicon
                import platform
                if platform.system() == 'Darwin' and platform.processor() == 'arm':
                    return "Apple Silicon GPU"
                    
                # Try checking for AMD GPU on Linux
                result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    return "AMD GPU"
                    
            except:
                pass
        
        return None
    
    def ensure_gpu_execution(self) -> bool:
        """Ensure model runs on GPU by checking and setting Ollama config."""
        try:
            # First check if we have GPU hardware
            if not self.gpu_info:
                logger.info("No GPU detected, will run on CPU")
                return False
            
            # Try to check current Ollama GPU status
            response = requests.get(f"{self.base_url}/api/show", headers=self.headers)
            if response.status_code == 200:
                config = response.json()
                if config.get("gpu", False):
                    logger.info("GPU already enabled in Ollama")
                    self.gpu_enabled = True
                    return True
            
            # Try to enable GPU through Ollama API
            response = requests.post(
                f"{self.base_url}/api/gpu",
                json={"enable": True},
                headers=self.headers
            )
            
            if response.status_code == 200:
                logger.info("Successfully enabled GPU in Ollama")
                self.gpu_enabled = True
                return True
            else:
                # If we can't enable GPU globally, we'll use per-request options
                logger.info("Will use per-request GPU options")
                self.gpu_enabled = True  # Still mark as enabled since we'll use per-request options
                return True
                
        except Exception as e:
            logger.warning(f"GPU configuration warning: {str(e)}")
            # If we have GPU hardware, still try to use it via per-request options
            if self.gpu_info:
                logger.info("Will attempt to use GPU via per-request options")
                self.gpu_enabled = True
                return True
            return False
    
    def _get_gpu_options(self) -> Dict[str, Any]:
        """Get GPU-specific options for model requests."""
        if not self.gpu_enabled:
            return {}
            
        options = {
            "num_gpu": 1,  # Start with 1 GPU by default
            "num_thread": 4,  # Conservative default
            "use_gpu": True
        }
        
        # Platform specific optimizations
        if 'NVIDIA' in str(self.gpu_info):
            options["num_thread"] = 8  # NVIDIA GPUs generally handle more threads well
        elif 'Apple Silicon' in str(self.gpu_info):
            options["num_thread"] = 4  # Conservative for Apple Silicon
            # Remove num_gpu for Apple Silicon as it's handled differently
            options.pop("num_gpu", None)
        
        return options
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name") for model in models]
            else:
                logger.error(f"Failed to get model list: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting model list: {str(e)}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama
        
        Args:
            model_name: Name of the model to pull
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                headers=self.headers,
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            logger.error(f"Error pulling model: {data['error']}")
                            return False
                        logger.info(f"Pull progress: {data.get('status', 'unknown')}")
                    except json.JSONDecodeError:
                        pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False
    
    def start_monitoring(self, estimated_tokens: Optional[int] = None):
        """Start monitoring token generation"""
        self.monitoring_active = True
        self.tokens_generated = 0
        self.token_timestamps = []
        self.generation_start_time = time.time()
        self.estimated_total_tokens = estimated_tokens or 500  # Use default if None
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=self._update_monitoring_stats)
        monitoring_thread.daemon = True
        monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring token generation"""
        self.monitoring_active = False
        time.sleep(0.3)  # Allow thread to exit
    
    def _update_monitoring_stats(self):
        """Update monitoring statistics in background thread"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                elapsed_time = current_time - self.generation_start_time
                
                # Calculate completion percentage
                if self.estimated_total_tokens > 0:  # This check is now safe
                    completion_percentage = min(100, (self.tokens_generated / self.estimated_total_tokens) * 100)
                    logger.info(f"Generation progress: {completion_percentage:.1f}%")
                
                # Calculate tokens per second
                if elapsed_time > 0:
                    tokens_per_second = self.tokens_generated / elapsed_time
                    logger.info(f"Generation speed: {tokens_per_second:.1f} tokens/sec")
                
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in monitoring thread: {str(e)}")
                time.sleep(0.5)  # Keep running even if there's an error
    
    def generate_text(self, 
                     prompt: str,
                     model: str = "qwen2.5:7b",
                     system_prompt: Optional[str] = None,
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     with_history: bool = True) -> Dict[str, Any]:
        """
        Generate text using Ollama model
        
        Args:
            prompt: User prompt
            model: Model name
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            with_history: Whether to include conversation history
            
        Returns:
            Dict containing generated text and metadata
        """
        try:
            # 检查基础URL是否设置
            if not self.base_url:
                logger.error("Ollama基础URL未设置")
                return {
                    "error": "Ollama基础URL未设置",
                    "text": "错误：Ollama服务URL未配置",
                    "source": "ollama"
                }
                
            # 检查model是否设置
            if not model:
                logger.error("Model名称未设置")
                return {
                    "error": "Model名称未设置",
                    "text": "错误：未指定模型名称",
                    "source": "ollama"
                }
                
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add conversation history if requested
            if with_history:
                messages.extend(self.conversation_history)
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Start monitoring
            self.start_monitoring(max_tokens)
            
            # Get GPU options
            options = {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
            # Add GPU options
            options.update(self._get_gpu_options())
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": options
                },
                headers=self.headers,
                stream=True
            )
            
            # Process streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "content" in data:
                            full_response += data["content"]
                            self.tokens_generated += 1
                            self.token_timestamps.append(time.time())
                    except json.JSONDecodeError:
                        pass
            
            # Stop monitoring
            self.stop_monitoring()
            
            # Update conversation history
            if with_history:
                self.conversation_history.append({
                    "role": "user",
                    "content": prompt
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Keep only last 10 messages
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
            
            return {
                "text": full_response,
                "model": model,
                "source": "ollama",
                "tokens_generated": self.tokens_generated,
                "processing_time": f"{time.time() - self.generation_start_time:.2f}秒"
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return {
                "error": str(e),
                "text": "Error generating response",
                "source": "ollama"
            }
    
    def generate_multimodal(self,
                          prompt: str,
                          image_path: Optional[str] = None,
                          image_base64: Optional[str] = None,
                          model: str = "llava:7b",
                          system_prompt: Optional[str] = None,
                          max_tokens: int = 1024,
                          temperature: float = 0.7,
                          top_p: float = 0.9) -> Dict[str, Any]:
        """
        Generate text using multimodal model with image input
        
        Args:
            prompt: User prompt
            image_path: Path to image file (optional)
            image_base64: Base64 encoded image (optional)
            model: Model name (default: llava:7b)
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Dict containing generated text and metadata
        """
        try:
            # Get image data
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
            elif image_base64:
                image_data = image_base64
            else:
                raise ValueError("Either image_path or image_base64 must be provided")
            
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add image and prompt
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "data": image_data
                    },
                    {
                        "type": "text",
                        "data": prompt
                    }
                ]
            })
            
            # Start monitoring
            self.start_monitoring(max_tokens)
            
            # Get GPU options
            options = {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
            # Add GPU options
            options.update(self._get_gpu_options())
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": options
                },
                headers=self.headers,
                stream=True
            )
            
            if response.status_code != 200:
                self.stop_monitoring()
                return {
                    "error": f"API request failed with status code: {response.status_code}",
                    "response": response.text
                }
            
            # Collect the full response while updating monitoring
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                            self.tokens_generated += 1
                            self.token_timestamps.append(time.time())
                    except json.JSONDecodeError:
                        pass
            
            # Stop monitoring
            self.stop_monitoring()
            
            # Store the conversation
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            return {
                "text": full_response,
                "model": model,
                "gpu_used": self.gpu_enabled,
                "tokens_generated": self.tokens_generated
            }
            
        except Exception as e:
            self.stop_monitoring()
            logger.error(f"Error in generate_multimodal: {str(e)}")
            return {"error": str(e)}
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = [] 