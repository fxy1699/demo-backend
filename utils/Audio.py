import edge_tts
import asyncio
from pydub import AudioSegment, effects
import simpleaudio as sa
import io
import numpy as np
import random
from scipy.signal import lfilter, firwin
import base64
import logging
import json
import requests
import os
import subprocess
import time
import threading
from typing import Iterator

# 智能频率适配参数
VOICE = "zh-CN-XiaoxiaoNeural"
MAX_FREQ_RATIO = 0.48  # 最大频率比例（Nyquist * 0.96）
CROSSFADE_DURATION = 500  # 交叉淡入淡出时长，单位为毫秒

# 保留原有函数 - 可能在其他地方被调用
def apply_tremolo(audio: AudioSegment, depth=30, freq=8):
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    channels = audio.channels
    frame_count = len(samples) // channels
    t = np.arange(frame_count) / sample_rate
    tremolo = 1 + (depth / 100.0) * np.sin(2 * np.pi * freq * t)

    if channels == 2:
        samples = samples.reshape((-1, 2))
        samples = (samples * tremolo[:, None]).astype(np.int16)
        samples = samples.flatten()
    else:
        samples = (samples * tremolo).astype(np.int16)

    return audio._spawn(samples.tobytes())

# 保留原有函数 - 可能在其他地方被调用
def pitch_shift(audio: AudioSegment, semitones):
    new_sample_rate = int(audio.frame_rate * (2.0 ** (semitones / 12.0)))
    shifted = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    return shifted.set_frame_rate(audio.frame_rate)

# 保留原有函数 - 可能在其他地方被调用
def apply_pitch_envelope(audio):
    chunks = []
    step = len(audio) // 6
    shifts = [+2, +3, +5, +5, +4, +5]  # 增加音调升降的幅度

    for i in range(6):
        part = audio[i * step: (i + 1) * step]
        shifted = pitch_shift(part, shifts[i])
        chunks.append(shifted)
    return sum(chunks)

# 保留原有函数 - 可能在其他地方被调用
def blend_with_parrot_chirp(voice_audio: AudioSegment, chirp_audio: AudioSegment):
    result = voice_audio
    duration = len(voice_audio)
    insert_times = list(range(0, duration - 1000, 1000))  # 每1秒插入一次

    chirp_insert_count = 0

    for t in insert_times:
        start = random.randint(0, max(0, len(chirp_audio) - 1200))
        chirp_snip = chirp_audio[start: start + 900]  # 限制叫声片段时长
        chirp_snip = chirp_snip - 10  # 降低叫声音量（重点调整：原来是 +6）

        # 轻混音，带 crossfade，避免突兀
        result = result.overlay(chirp_snip, position=t, gain_during_overlay=-3)
        chirp_insert_count += 1

    print(f"共插入了 {chirp_insert_count} 段调整音量后的鹦鹉叫声")
    return result

# 改进自适应语音清晰度处理函数，接受文本参数
async def adaptive_voice_clarity(text, pitch="+600Hz", rate="+5%", volume="+5%"):
    """
    使用自适应滤波器处理语音，提高清晰度
    
    参数:
        text (str): 要转换为语音的文本
        pitch (str): 音高调整，默认 +600Hz
        rate (str): 语速调整，默认 +5%
        volume (str): 音量调整，默认 +5%
        
    返回:
        AudioSegment: 处理后的音频段
    """
    # 生成原始语音
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        pitch=pitch,
        rate=rate,
        volume=volume
    )
    
    # 采集音频字节
    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])
    
    # 加载并标准化音频
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3").set_frame_rate(24000)  # 固定采样率
    samples = np.array(audio.get_array_of_samples())
    sr = audio.frame_rate
    nyquist = sr // 2  # 计算奈奎斯特频率

    # 自动频率适配带通滤波器
    def auto_bandpass(samples, sr):
        # 根据采样率动态设置频段
        freq_bands = [
            (200, min(800, nyquist * 0.8)),
            (min(800, nyquist * 0.4), min(3000, nyquist * 0.8)),
            (min(3000, nyquist * 0.6), min(nyquist * MAX_FREQ_RATIO, 7000))
        ]

        processed = np.zeros_like(samples, dtype=np.float64)
        for low, high in freq_bands:
            # 跳过无效频段
            if high <= low or high >= nyquist:
                continue

            # 设计滤波器
            taps = firwin(
                numtaps=301,
                cutoff=[low, high],
                fs=sr,
                pass_zero=False
            )
            # 分频段处理
            band = lfilter(taps, 1.0, samples)
            processed += band * (1 + 0.2 * np.log(high / 1000))  # 动态增益

        return processed.astype(np.int16)

    processed = auto_bandpass(samples, sr)

    # 构建专业音频
    final_audio = AudioSegment(
        processed.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    return final_audio

# 改进音频拼接函数，接受已处理的音频和鹦鹉叫声路径
async def join_audio(processed_audio, parrot_sound_path="./utils/audio_files/new_parrot_chirp_1.wav"):
    """
    将处理后的语音与鹦鹉叫声拼接
    
    参数:
        processed_audio (AudioSegment): 已处理的语音段
        parrot_sound_path (str): 鹦鹉叫声音频文件路径
        
    返回:
        AudioSegment: 拼接后的音频段
    """
    try:
        # 加载要拼接的鹦鹉音频
        parrot_audio = AudioSegment.from_wav(parrot_sound_path).set_frame_rate(24000)
    except Exception as e:
        logging.warning(f"错误：无法加载鹦鹉叫声音频文件: {parrot_sound_path}. 错误信息: {e}")
        return processed_audio

    # 随机截取一秒左右的音频
    one_second = 3000  # 1秒 = 1000毫秒
    if len(parrot_audio) > one_second:
        start_time = random.randint(0, len(parrot_audio) - one_second)
        end_time = start_time + one_second
        parrot_audio_clip = parrot_audio[start_time:end_time]
    else:
        parrot_audio_clip = parrot_audio

    # 进行交叉淡入淡出拼接
    if len(processed_audio) >= CROSSFADE_DURATION and len(parrot_audio_clip) >= CROSSFADE_DURATION:
        joined_audio = processed_audio.fade(to_gain=-120, start=len(processed_audio) - CROSSFADE_DURATION, duration=CROSSFADE_DURATION)
        joined_audio = joined_audio.append(parrot_audio_clip.fade(from_gain=-120, start=0, duration=CROSSFADE_DURATION), crossfade=CROSSFADE_DURATION)
    else:
        joined_audio = processed_audio + parrot_audio_clip

    return joined_audio

# 更新后的主函数 - 使用新的处理方法
async def tts_real_parrot_play(text):
    """
    将文本转换为语音，并播放（带鹦鹉叫声）
    
    参数:
        text (str): 要转换为语音的文本
    
    返回:
        AudioSegment: 处理后的音频段
    """
    # 使用自适应语音清晰度处理生成语音
    processed_audio = await adaptive_voice_clarity(text)
    
    # 拼接鹦鹉叫声
    final_audio = await join_audio(processed_audio, "./utils/audio_files/new_parrot_chirp_1.wav")
    
    # 播放
    play_obj = sa.play_buffer(
        final_audio.raw_data,
        num_channels=final_audio.channels,
        bytes_per_sample=final_audio.sample_width,
        sample_rate=final_audio.frame_rate
    )
    play_obj.wait_done()
    
    # 返回处理后的音频
    return final_audio

# 给后端 API 使用的函数
async def _text_to_speech_core(text, style='cartoon'):
    """
    核心TTS处理函数
    
    参数:
        text (str): 要转换的文本
        style (str): 语音风格，默认为 'cartoon'
        
    返回:
        BytesIO: 包含处理后音频的字节流
    """
    if style == 'cartoon':
        # 使用新的处理方法
        processed_audio = await adaptive_voice_clarity(text)
        final_audio = await join_audio(processed_audio)
    else:
        # 使用原始 edge_tts
        voice = "zh-CN-YunxiNeural" if style != 'cartoon' else VOICE
        communicate = edge_tts.Communicate(text=text, voice=voice)
        audio_bytes = bytearray()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
        
        final_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    
    # 转换为 IO 对象
    output = io.BytesIO()
    final_audio.export(output, format="wav")
    output.seek(0)
    
    return output

def upload_audio_for_voice_clone(audio_file_path):
    """
    上传音频文件用于语音克隆
    
    参数:
        audio_file_path (str): 音频文件路径
        
    返回:
        str: 上传成功后的文件ID
    """
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            group_id = '1913402932208866274'
            api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w'
            
            url = f'https://api.minimaxi.chat/v1/files/upload?GroupId={group_id}'
            headers = {
                'authority': 'api.minimaxi.chat',
                'Authorization': f'Bearer {api_key}'
            }
            
            data = {
                'purpose': 'voice_clone'
            }
            
            if not os.path.exists(audio_file_path):
                logging.error(f"语音克隆音频文件不存在: {audio_file_path}")
                return None
                
            # 检查文件大小
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                logging.error(f"语音克隆音频文件为空: {audio_file_path}")
                return None
                
            if file_size > 5 * 1024 * 1024:  # 大于5MB
                logging.warning(f"语音克隆音频文件较大({file_size/1024/1024:.2f}MB)，可能导致上传超时")
                
            logging.info(f"开始上传音频文件: {audio_file_path}, 大小: {file_size/1024:.2f}KB")
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': audio_file
                }
                response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
                
            if response.status_code == 200:
                response_data = response.json()
                file_id = response_data.get("file", {}).get("file_id")
                if file_id:
                    logging.info(f"成功上传音频文件 {os.path.basename(audio_file_path)}, 文件ID: {file_id}")
                    return file_id
                else:
                    logging.error(f"上传成功但未获取到文件ID，响应: {response_data}")
            else:
                logging.error(f"上传音频文件失败: HTTP {response.status_code}, {response.text}")
                
            # 如果失败但不是因为网络问题，不重试
            if response.status_code != 500 and response.status_code != 503 and response.status_code != 504:
                return None
                
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"尝试第 {retry_count} 次重试上传...")
                time.sleep(1)  # 延迟1秒后重试
                
        except requests.exceptions.RequestException as e:
            logging.error(f"上传音频文件网络异常: {str(e)}")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"尝试第 {retry_count} 次重试上传...")
                time.sleep(1)  # 延迟1秒后重试
            else:
                break
        except Exception as e:
            logging.error(f"上传音频文件未知异常: {str(e)}")
            return None
            
    logging.error(f"上传音频文件失败，已重试 {max_retries} 次")
    return None

def clone_voice(file_id, voice_id="ppooiudiii"):
    """
    使用已上传的音频文件克隆语音
    
    参数:
        file_id (str): 上传的音频文件ID
        voice_id (str): 要使用的语音ID
        
    返回:
        bool: 克隆是否成功
    """
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            group_id = '1913402932208866274'
            api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w'
            
            url = f'https://api.minimaxi.chat/v1/voice_clone?GroupId={group_id}'
            payload = json.dumps({
                "file_id": file_id,
                "voice_id": voice_id
            })
            headers = {
                'authorization': f'Bearer {api_key}',
                'content-type': 'application/json'
            }
            
            logging.info(f"开始克隆语音: file_id={file_id}, voice_id={voice_id}")
            response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success", False):
                    logging.info(f"语音克隆成功，响应: {response_data}")
                    return True
                else:
                    error_msg = response_data.get("error", {}).get("message", "未知错误")
                    logging.error(f"语音克隆失败，错误信息: {error_msg}")
                    # 如果是参数错误，不再重试
                    if "invalid" in error_msg.lower() or "not found" in error_msg.lower():
                        return False
            else:
                logging.error(f"语音克隆API调用失败: HTTP {response.status_code}, {response.text}")
                
            # 如果失败但不是因为网络问题，不重试
            if response.status_code != 500 and response.status_code != 503 and response.status_code != 504:
                return False
                
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"尝试第 {retry_count} 次重试克隆...")
                time.sleep(1)  # 延迟1秒后重试
                
        except requests.exceptions.RequestException as e:
            logging.error(f"语音克隆网络异常: {str(e)}")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"尝试第 {retry_count} 次重试克隆...")
                time.sleep(1)  # 延迟1秒后重试
            else:
                break
        except Exception as e:
            logging.error(f"语音克隆未知异常: {str(e)}")
            return False
            
    logging.error(f"语音克隆失败，已重试 {max_retries} 次")
    return False

def text_to_speech_base64(text, style='minimax', voice_id="ppooiudiii", custom_audio_path=None, speed=1.0, volume=1.0, pitch=0):
    """
    将文本转换为语音并返回 base64 编码
    
    参数:
        text (str): 要转换的文本
        style (str): 语音风格，默认为 'minimax'，其他选项: 'cartoon'
        voice_id (str): Minimax的语音ID，默认为"ppooiudiii"
        custom_audio_path (str): 用于语音克隆的音频文件路径，默认为"i.wav"
        speed (float): 语音速度，范围0.5-2.0
        volume (float): 语音音量，范围0.5-2.0
        pitch (int): 音调调整，范围-12到12
        
    返回:
        str: 音频的 base64 编码，如果失败则返回 None
    """
    try:
        logging.info(f"开始文本转语音: style={style}, voice_id={voice_id}, custom_audio_path={custom_audio_path}")
        
        if style == 'minimax':
            # Minimax TTS API 配置
            group_id = '1913402932208866274'
            api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w'
            url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={group_id}"
            headers = {
                'accept': 'application/json, text/plain, */*',
                'content-type': 'application/json',
                'authorization': f"Bearer {api_key}",
            }

            # 如果提供了自定义音频文件，先上传并克隆语音
            file_id = None
            clone_success = False
            
            if custom_audio_path is None:
                custom_audio_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "i.wav")
            
            if not os.path.exists(custom_audio_path):
                logging.error(f"语音克隆音频文件不存在: {custom_audio_path}")
            else:
                logging.info(f"找到音频文件用于克隆: {custom_audio_path}")
                # 上传音频文件
                file_id = upload_audio_for_voice_clone(custom_audio_path)
                
                if file_id:
                    logging.info(f"文件上传成功，ID: {file_id}")
                    # 进行语音克隆
                    clone_success = clone_voice(file_id, voice_id)
                    
                    if clone_success:
                        logging.info(f"语音克隆成功，将使用克隆后的音色: {voice_id}")
                    else:
                        logging.warning(f"语音克隆失败，将使用默认语音 {voice_id}")
                else:
                    logging.error(f"文件上传失败，将使用默认语音")
            
            # 构建请求体
            body = json.dumps({
                "model": "speech-02-turbo",
                "text": text,
                "stream": False,  # 非流式请求以获取完整音频
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": speed,
                    "vol": volume,
                    "pitch": pitch
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1
                }
            })
            
            # 发送请求获取音频
            logging.info(f"发送Minimax TTS请求: 文本长度={len(text)}, 音色ID={voice_id}")
            try:
                response = requests.request("POST", url, headers=headers, data=body)
                logging.info(f"Minimax TTS响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    if "data" in response_data and "audio" in response_data["data"]:
                        # 解码音频数据并转换为base64
                        logging.info("成功获取Minimax音频数据")
                        audio_hex = response_data["data"]["audio"]
                        audio_bytes = bytes.fromhex(audio_hex)
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        return audio_base64
                    else:
                        logging.error(f"Minimax TTS 返回数据结构异常: {response_data}")
                else:
                    logging.error(f"Minimax TTS 请求失败: {response.status_code}, {response.text}")
            except Exception as inner_e:
                logging.error(f"Minimax TTS 请求异常: {str(inner_e)}")
                
            # 如果 Minimax TTS 失败，回退到默认方法
            logging.warning("Minimax TTS 失败，回退到默认方法")
            
        # 使用原有的方法
        logging.info(f"使用原有的文本转语音方法: style={style}")
        audio_io = asyncio.run(_text_to_speech_core(text, style if style != 'minimax' else 'cartoon'))
        audio_data = audio_io.getvalue()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        logging.info("成功生成原有方法的音频数据")
        return audio_base64
    except Exception as e:
        logging.error(f"文本转语音(base64)失败: {str(e)}")
        return None

def text_to_speech_bytes(text, style='minimax', voice_id="ppooiudiii", custom_audio_path=None, speed=1.0, volume=1.0, pitch=0):
    """
    将文本转换为语音并返回字节
    
    参数:
        text (str): 要转换的文本
        style (str): 语音风格，默认为 'minimax'，其他选项: 'cartoon'
        voice_id (str): Minimax的语音ID，默认为"ppooiudiii"
        custom_audio_path (str): 用于语音克隆的音频文件路径
        speed (float): 语音速度，范围0.5-2.0
        volume (float): 语音音量，范围0.5-2.0
        pitch (int): 音调调整，范围-12到12
        
    返回:
        BytesIO: 包含音频的字节流，如果失败则返回None
    """
    try:
        if style == 'minimax':
            # 使用 text_to_speech_base64 获取 base64 编码的音频
            audio_base64 = text_to_speech_base64(
                text=text, 
                style=style, 
                voice_id=voice_id, 
                custom_audio_path=custom_audio_path, 
                speed=speed, 
                volume=volume, 
                pitch=pitch
            )
            if audio_base64:
                # 将 base64 解码为字节
                audio_bytes = base64.b64decode(audio_base64)
                # 创建 BytesIO 对象
                audio_io = io.BytesIO(audio_bytes)
                audio_io.seek(0)
                return audio_io
            return None
        
        # 使用原有的方法
        audio_io = asyncio.run(_text_to_speech_core(text, style))
        return audio_io
    except Exception as e:
        logging.error(f"文本转语音(bytes)失败: {str(e)}")
        return None

# 测试函数
async def test_improved_tts():
    """测试自适应语音清晰度和拼接功能"""
    text = "你好呀~我是聪明的小鹦鹉！"
    
    # 生成并处理语音
    processed_audio = await adaptive_voice_clarity(text)
    processed_audio.export("adaptive_voice.wav", format="wav")
    print("自适应语音处理完成，已保存为 adaptive_voice.wav")
    
    # 拼接鹦鹉叫声
    final_audio = await join_audio(processed_audio)
    final_audio.export("joined_voice.wav", format="wav")
    print("音频拼接完成，已保存为 joined_voice.wav")

# 示例运行
if __name__ == "__main__":
    # asyncio.run(test_improved_tts())
    text = "别难过啦！有我陪着你呢，你可以和我说说呀~"
    asyncio.run(tts_real_parrot_play(text))

def get_minimax_tts_stream(text, voice_id="ppooiudiii", speed=1.0, volume=1.0, pitch=0) -> Iterator[bytes]:
    """
    获取Minimax文本转语音流式数据
    
    参数:
        text (str): 要转换的文本
        voice_id (str): Minimax的语音ID，默认为"ppooiudiii"
        speed (float): 语音速度，范围0.5-2.0
        volume (float): 语音音量，范围0.5-2.0
        pitch (int): 音调调整，范围-12到12
        
    返回:
        Iterator[bytes]: 音频数据流
    """
    try:
        group_id = '1913402932208866274'
        api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w'
        url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={group_id}"
        
        headers = {
            'accept': 'application/json, text/plain, */*',
            'content-type': 'application/json',
            'authorization': f"Bearer {api_key}",
        }
        
        body = json.dumps({
            "model": "speech-02-turbo",
            "text": text,
            "stream": True,  # 流式请求
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": volume,
                "pitch": pitch
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            }
        })
        
        response = requests.request("POST", url, stream=True, headers=headers, data=body)
        
        if response.status_code != 200:
            logging.error(f"Minimax TTS流式请求失败: {response.status_code}, {response.text}")
            return
            
        for chunk in response.raw:
            if chunk:
                if chunk[:5] == b'data:':
                    data = json.loads(chunk[5:])
                    if "data" in data and "extra_info" not in data:
                        if "audio" in data["data"]:
                            audio = data["data"]['audio']
                            yield audio
    except Exception as e:
        logging.error(f"获取Minimax TTS流失败: {str(e)}")

def play_audio_stream_with_mpv(audio_stream: Iterator[bytes]) -> bytes:
    """
    使用MPV播放音频流
    
    参数:
        audio_stream (Iterator[bytes]): 音频数据流
        
    返回:
        bytes: 完整的音频数据
    """
    try:
        # 使用相对路径
        mpv_exe_path = "./utils/mpv.exe"
        
        mpv_command = [mpv_exe_path, "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        audio = b""
        for chunk in audio_stream:
            if chunk is not None and chunk != '\n':
                try:
                    decoded_hex = bytes.fromhex(chunk)
                    if mpv_process.stdin:
                        mpv_process.stdin.write(decoded_hex)
                        mpv_process.stdin.flush()
                    audio += decoded_hex
                except Exception as e:
                    logging.error(f"音频流处理错误: {str(e)}")
        
        return audio
    except Exception as e:
        logging.error(f"播放音频流失败: {str(e)}")
        return b""

def text_to_speech_stream_play(text, voice_id="ppooiudiii", speed=1.0, volume=1.0, pitch=0, save_to_file=False):
    """
    将文本转换为语音并进行流式播放
    
    参数:
        text (str): 要转换的文本
        voice_id (str): Minimax的语音ID，默认为"ppooiudiii"
        speed (float): 语音速度，范围0.5-2.0
        volume (float): 语音音量，范围0.5-2.0
        pitch (int): 音调调整，范围-12到12
        save_to_file (bool): 是否保存到文件
        
    返回:
        bytes: 音频数据
    """
    try:
        audio_stream = get_minimax_tts_stream(text, voice_id, speed, volume, pitch)
        audio = play_audio_stream_with_mpv(audio_stream)
        
        # 保存结果到文件
        if save_to_file and audio:
            timestamp = int(time.time())
            file_name = f'output_stream_{timestamp}.mp3'
            with open(file_name, 'wb') as file:
                file.write(audio)
            logging.info(f"已保存音频到文件: {file_name}")
            
        return audio
    except Exception as e:
        logging.error(f"流式文本转语音播放失败: {str(e)}")
        return b""

# 增加异步播放函数，不阻塞主线程
def text_to_speech_stream_play_async(text, voice_id="ppooiudiii", speed=1.0, volume=1.0, pitch=0, save_to_file=False):
    """
    异步将文本转换为语音并进行流式播放
    
    参数:
        text (str): 要转换的文本
        voice_id (str): Minimax的语音ID，默认为"ppooiudiii"
        speed (float): 语音速度，范围0.5-2.0
        volume (float): 语音音量，范围0.5-2.0
        pitch (int): 音调调整，范围-12到12
        save_to_file (bool): 是否保存到文件
    """
    def _play_in_thread():
        text_to_speech_stream_play(text, voice_id, speed, volume, pitch, save_to_file)
    
    thread = threading.Thread(target=_play_in_thread)
    thread.daemon = True
    thread.start()
    return thread
