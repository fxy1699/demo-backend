import cv2
import numpy as np
from deepface import DeepFace
import collections
import os
import json
import time

# 移除摄像头相关代码和线程处理代码

def analyze_emotion_from_image(image_data):
    """
    从图像数据中分析情绪
    
    参数:
        image_data: 图像数据（numpy数组或字节数据）
    
    返回:
        dict: 包含分析结果的字典，包括情绪和置信度
    """
    try:
        # 如果输入是字节数据，转换为numpy数组
        if isinstance(image_data, bytes):
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            image = image_data
            
        # 调整图像大小以加快处理速度
        small_frame = cv2.resize(image, (320, 240))
        
        # 分析情绪
        results = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        
        if results:
            # 提取主要情绪和置信度
            detected_emotion = results[0]['dominant_emotion']
            emotion_scores = results[0]['emotion']
            
            # 将情绪映射为前端可接受的类型
            mapped_emotion = map_emotion_to_frontend(detected_emotion)
            
            return {
                'status': 'success',
                'emotion': mapped_emotion,
                'original_emotion': detected_emotion,
                'confidence': emotion_scores[detected_emotion],
                'all_emotions': emotion_scores
            }
        else:
            return {
                'status': 'error',
                'message': '无法检测到面部表情',
                'emotion': 'neutral'  # 默认情绪
            }
            
    except Exception as e:
        print(f"分析情绪时出错: {str(e)}")
        return {
            'status': 'error',
            'message': f'分析失败: {str(e)}',
            'emotion': 'neutral'  # 默认情绪
        }

def map_emotion_to_frontend(emotion):
    """
    将DeepFace检测到的情绪映射为前端可接受的类型
    
    参数:
        emotion: DeepFace检测到的情绪
        
    返回:
        str: 映射后的情绪类型 (happy, sad, angry, neutral)
    """
    # 情绪映射表
    emotion_map = {
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'disgust': 'angry',  # 将厌恶映射为愤怒
        'fear': 'sad',       # 将恐惧映射为悲伤
        'surprise': 'happy', # 将惊讶映射为快乐
        'neutral': 'neutral'
    }
    
    return emotion_map.get(emotion, 'neutral')

# 移除其他不需要的函数和代码