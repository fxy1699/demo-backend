# This Python file uses the following encoding: utf-8

import json
import time
import tempfile
import os
import base64
from typing import Iterator, Optional, Dict, Any
import pygame
import requests
import traceback

# Minimax API配置
GROUP_ID = '1913402932208866274'    # your_group_id
API_KEY = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w'

# 文件格式配置
FILE_FORMAT = 'mp3'  # support mp3/pcm/flac

# 初始化pygame
try:
    pygame.mixer.init()
except:
    print("警告：pygame初始化失败，音频播放功能可能不可用")

# 保存克隆语音ID
CLONED_VOICE_ID = None
VOICE_FILE_PATH = None

class MinimaxTTS:
    def __init__(self, api_key: str, group_id: str):
        self.api_key = api_key
        self.group_id = group_id
        self.base_url = "https://api.minimax.chat/v1"
        self.voice_file_path = None
        self.cloned_voice_id = None
        
    def set_voice_file_path(self, file_path: str) -> None:
        """设置用于克隆的语音文件路径"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"语音文件不存在: {file_path}")
        self.voice_file_path = file_path
        
    def clone_voice(self) -> str:
        """克隆语音并返回voice_id"""
        if not self.voice_file_path:
            raise ValueError("请先设置语音文件路径")
            
        # 上传文件
        upload_url = f"{self.base_url}/files/upload?GroupId={self.group_id}"
        headers = {
            'authority': 'api.minimax.chat',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'purpose': 'voice_clone'
        }
        
        with open(self.voice_file_path, 'rb') as f:
            files = {
                'file': f
            }
            print("正在上传语音文件...")
            response = requests.post(upload_url, headers=headers, data=data, files=files)
            
        if response.status_code != 200:
            raise Exception(f"文件上传失败: {response.text}")
            
        result = response.json()
        if "file" not in result:
            raise Exception(f"上传响应异常: {result}")
            
        file_id = result["file"].get("file_id")
        if not file_id:
            raise Exception("未获取到file_id")
            
        # 执行克隆
        clone_url = f"{self.base_url}/voice_clone?GroupId={self.group_id}"
        unique_voice_id = f"voice_{int(time.time())}"
        payload = json.dumps({
            "file_id": file_id,
            "voice_id": unique_voice_id
        })
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'content-type': 'application/json'
        }
        
        print(f"正在执行语音克隆，使用voice_id: {unique_voice_id}...")
        response = requests.post(clone_url, headers=headers, data=payload)
        
        if response.status_code != 200:
            raise Exception(f"语音克隆失败: {response.text}")
            
        result = response.json()
        print("克隆响应：", result)
        
        self.cloned_voice_id = unique_voice_id
        return unique_voice_id

    def text_to_speech(self, text: str, output_path: str, voice_id: Optional[str] = None) -> None:
        """将文本转换为语音"""
        if not voice_id and not self.cloned_voice_id:
            raise ValueError("请先克隆语音或提供voice_id")
            
        url = f"{self.base_url}/text_to_speech?GroupId={self.group_id}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "text": text,
            "voice_id": voice_id or self.cloned_voice_id,
            "model": "speech-01",
            "speed": 1.0
        }
        
        print(f"正在生成语音，使用voice_id: {voice_id or self.cloned_voice_id}...")
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"语音生成失败: {response.text}")
            
        # 保存音频文件
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"语音已保存到: {output_path}")

def get_voice_id():
    """获取当前使用的语音ID"""
    global CLONED_VOICE_ID
    
    # 如果没有克隆的语音ID，执行克隆流程
    if CLONED_VOICE_ID is None:
        try:
            CLONED_VOICE_ID = clone_voice_from_file()
        except Exception as e:
            print(f"语音克隆失败，使用默认语音ID: {e}")
            CLONED_VOICE_ID = "ppooiudiii"  # 使用默认语音
    
    return CLONED_VOICE_ID

def build_tts_stream_headers() -> dict:
    """构建TTS流式请求的头部"""
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'authorization': "Bearer " + API_KEY,
    }
    return headers

def build_tts_stream_body(text: str, style: str = 'default') -> dict:
    """构建TTS流式请求的请求体，使用克隆的语音ID"""
    voice_id = get_voice_id()
    
    # 根据风格调整语音参数
    if style == 'cartoon':
        speed = 0.8  # 降低卡通风格的语速
        vol = 1.0
        pitch = 2
    elif style == 'slow':
        speed = 0.5  # 更慢的语速
        vol = 1.0
        pitch = -1
    elif style == 'very_slow':
        speed = 0.4  # 非常慢的语速
        vol = 1.0
        pitch = -1
    else:  # 默认风格
        speed = 0.6  # 降低默认语速
        vol = 1.0
        pitch = 0
    
    body = json.dumps({
        "model": "speech-02-turbo",
        "text": text,
        "stream": True,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "vol": vol,
            "pitch": pitch
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": FILE_FORMAT,
            "channel": 1
        }
    })
    return body

def call_tts_stream(text: str, style: str = 'default') -> Iterator[bytes]:
    """调用TTS流式API并返回音频流迭代器"""
    tts_url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={GROUP_ID}"
    tts_headers = build_tts_stream_headers()
    tts_body = build_tts_stream_body(text, style)

    print(f"调用Minimax TTS API，文本: '{text[:30]}...'，风格: {style}")
    try:
        response = requests.request("POST", tts_url, stream=True, headers=tts_headers, data=tts_body)
        
        if response.status_code != 200:
            print(f"TTS API返回错误状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return
        
        count = 0
        buffer = []
        for chunk in response.raw:
            if chunk:
                if chunk[:5] == b'data:':
                    try:
                        data = json.loads(chunk[5:])
                        if "data" in data and "extra_info" not in data:
                            if "audio" in data["data"]:
                                audio = data["data"]['audio']
                                count += 1
                                # 立即解码16进制数据为字节
                                decoded_hex = bytes.fromhex(audio)
                                buffer.append(decoded_hex)
                                # 每收集到3个块就yield一次，减少延迟
                                if len(buffer) >= 3:
                                    yield b''.join(buffer)
                                    buffer = []
                    except json.JSONDecodeError as e:
                        print(f"解析JSON失败: {e}, 数据: {chunk[:50]}...")
        
        # 发送剩余的音频数据
        if buffer:
            yield b''.join(buffer)
            
        print(f"TTS API调用完成，成功接收 {count} 个音频块")
    except Exception as e:
        print(f"TTS API调用失败: {e}")
        traceback.print_exc()

def audio_play(audio_stream: Iterator[bytes]) -> bytes:
    """接收音频流并播放，同时返回完整的音频数据"""
    audio = b""
    
    # 收集所有音频数据
    for chunk in audio_stream:
        if chunk is not None and chunk != '\n':
            decoded_hex = bytes.fromhex(chunk)
            audio += decoded_hex
    
    # 创建临时文件用于播放
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{FILE_FORMAT}') as temp_file:
        temp_file.write(audio)
        temp_path = temp_file.name
    
    # 使用pygame播放音频
    try:
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # 等待音频播放完成
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"播放音频失败: {e}")
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
    
    return audio

# 以下是与原有Audio.py接口兼容的函数

def text_to_speech_base64(text: str, style: str = 'default') -> Optional[str]:
    """将文本转换为base64编码的音频数据，使用克隆的语音"""
    try:
        # 使用更高效的方式收集音频数据
        audio_data = b""
        audio_chunk_iterator = call_tts_stream(text, style)
        
        # 预分配更大的缓冲区以减少字节串连接
        chunks = []
        for chunk in audio_chunk_iterator:
            if chunk is not None and chunk != '\n':
                chunks.append(chunk)
        
        # 一次性合并所有块
        if chunks:
            audio_data = b''.join(chunks)
            print(f"成功生成音频数据，大小: {len(audio_data)} 字节")
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            return audio_base64
        
        print("未能生成音频数据")
        return None
    except Exception as e:
        print(f"文本转语音失败: {e}")
        traceback.print_exc()
        return None

def text_to_speech_bytes(text: str, style: str = 'default') -> Optional[bytes]:
    """将文本转换为音频字节数据，使用克隆的语音"""
    try:
        print(f"将文本转换为音频: '{text[:30]}...'，风格: {style}")
        
        # 使用更高效的方式收集音频数据
        audio_chunk_iterator = call_tts_stream(text, style)
        
        # 预分配更大的缓冲区以减少字节串连接
        chunks = []
        
        # 这里音频块已经是解码后的bytes，不需要再次解码
        for chunk in audio_chunk_iterator:
            if chunk is not None and chunk != b'\n' and len(chunk) > 0:
                chunks.append(chunk)
        
        # 一次性合并所有块
        if chunks:
            audio_data = b''.join(chunks)
            print(f"成功生成音频数据，大小: {len(audio_data)} 字节")
            return audio_data
        
        print("未能生成任何音频数据块")
        return None
    except Exception as e:
        print(f"文本转语音失败: {e}")
        traceback.print_exc()
        return None

def tts_real_parrot_play(text: str, style: str = 'default') -> Optional[str]:
    """将文本转换为语音并播放，使用克隆的语音，返回文件路径"""
    try:
        # 使用我们优化过的字节处理函数
        audio_data = text_to_speech_bytes(text, style)
        
        if not audio_data:
            print("无法生成音频数据")
            return None
        
        # 保存音频文件
        timestamp = int(time.time())
        file_name = f'output_{timestamp}.{FILE_FORMAT}'
        with open(file_name, 'wb') as file:
            file.write(audio_data)
        
        # 使用pygame播放音频
        try:
            pygame.mixer.music.load(file_name)
            pygame.mixer.music.play()
            
            # 等待音频播放完成
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        except Exception as e:
            print(f"播放音频失败: {e}")
        
        return file_name
    except Exception as e:
        print(f"文本转语音并播放失败: {e}")
        return None

# 以下是占位函数，确保与原有接口兼容

def apply_tremolo(audio_data, rate=6.0, depth=0.5):
    """应用颤音效果（占位函数）"""
    return audio_data

def pitch_shift(audio_data, semitones=2):
    """改变音高（占位函数）"""
    return audio_data

def apply_pitch_envelope(audio_data, envelope=None):
    """应用音高包络（占位函数）"""
    return audio_data

def blend_with_parrot_chirp(audio_data, chirp_file=None, blend_factor=0.3):
    """与鹦鹉声混合（占位函数）"""
    return audio_data

def adaptive_voice_clarity(audio_data, clarity_level=1.0):
    """自适应语音清晰度（占位函数）"""
    return audio_data

def join_audio(audio_files):
    """连接多个音频文件（占位函数）"""
    return None

# 添加一个新的流式函数
def text_to_speech_stream(text: str, style: str = 'default') -> Iterator[bytes]:
    """
    将文本转换为语音流，直接返回每个音频块，无需等待全部数据

    参数:
        text (str): 要转换的文本
        style (str): 语音风格 ('default', 'cartoon', 'slow', 'very_slow')
        
    返回:
        Iterator[bytes]: 音频块迭代器，每个块是解码后的音频数据
    """
    try:
        # 调用 TTS 流式 API
        audio_chunk_iterator = call_tts_stream(text, style)
        
        # 直接返回解码后的块，不累积完整音频
        for chunk in audio_chunk_iterator:
            if chunk is not None and chunk != '\n':
                # 将十六进制字符串解码为字节
                decoded_hex = bytes.fromhex(chunk)
                yield decoded_hex
                
    except Exception as e:
        print(f"流式文本转语音失败: {e}")
        yield b""  # 发生错误时返回空字节

# 在模块导入时执行语音克隆
print("初始化语音克隆模块...")
# 如果您有指定的语音文件路径，可以在此处设置
# set_voice_file_path("path/to/your/voice/file.wav")

# 简单测试
if __name__ == "__main__":
    # 执行语音克隆
    try:
        voice_id = clone_voice_from_file()
        print(f"测试语音克隆成功，voice_id: {voice_id}")
        
        # 测试文本转语音
        text = "你好，我是使用克隆语音的助手。"
        audio_file = tts_real_parrot_play(text)
        print(f"音频文件已保存至: {audio_file}")
    except Exception as e:
        print(f"测试失败: {e}")

    # 使用新的MinimaxTTS类进行测试
    try:
        api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w"
        group_id = "1913402932208866274"
        
        tts = MinimaxTTS(api_key, group_id)
        
        # 设置语音文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_file = os.path.join(os.path.dirname(script_dir), "i.wav")
        tts.set_voice_file_path(default_file)
        
        # 克隆语音
        voice_id = tts.clone_voice()
        print(f"语音克隆成功，voice_id: {voice_id}")
        
        # 使用克隆的语音生成文本
        output_path = os.path.join(os.path.dirname(script_dir), "output.wav")
        tts.text_to_speech("你好，这是一个测试。", output_path)
        
    except Exception as e:
        print(f"测试失败: {str(e)}") 