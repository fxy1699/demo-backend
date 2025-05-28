"""
情感思维链(ECoT)处理模块
该模块实现了图片中所示的ECoT框架，包括理解上下文、识别他人情绪、识别自我情绪、
管理自我情绪、影响他人情绪，然后生成情感响应。
"""

import json
import logging
from typing import Dict, Any, List, Optional

from utils.prompt_manager import get_emotion_prompt

# 配置日志
logger = logging.getLogger(__name__)

class ECoTHandler:
    """情感思维链处理器，实现情感共情响应生成"""
    
    def __init__(self, llm_handler=None):
        """
        初始化情感思维链处理器
        
        参数:
            llm_handler: LLM处理器实例
        """
        self.llm_handler = llm_handler
    
    def generate_empathetic_response(self, 
                                    query: str, 
                                    context: Dict[str, Any],
                                    emotion_condition: str = "empathy",
                                    system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        生成情感共情响应
        
        参数:
            query: 用户查询
            context: 对话上下文
            emotion_condition: 情绪条件（如empathy、sympathy等）
            system_prompt: 自定义系统提示
            
        返回:
            包含ECoT过程和最终响应的结果
        """
        if not self.llm_handler:
            return self._mock_empathetic_response(query, context, emotion_condition)
        
        # 构建ECoT提示
        prompt = self._build_ecot_prompt(query, context, emotion_condition)
        
        # 如果没有提供系统提示，使用情感提示中的"text"类型
        if not system_prompt:
            system_prompt = get_emotion_prompt('text')
        
        # 使用LLM生成响应
        result = self.llm_handler.generate_text(prompt, system_prompt, max_tokens=1024)
        
        try:
            # 解析LLM响应
            parsed_result = self._parse_ecot_response(result["text"])
            
            # 添加处理时间
            parsed_result["processing_time"] = result.get("processing_time", "未知")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"解析ECoT响应时出错: {str(e)}")
            return {
                "step1_context": f"分析失败: {str(e)}",
                "step2_others_emotions": "未知",
                "step3_self_emotions": "未知",
                "step4_managing_emotions": "未知", 
                "step5_influencing_emotions": "未知",
                "response": "很抱歉，我无法正确理解您的情绪。",
                "emotion": "neutral",
                "error": str(e)
            }
    
    def _build_ecot_prompt(self, query: str, context: Dict[str, Any], emotion_condition: str) -> str:
        """构建ECoT提示"""
        
        # 提取听众信息和上下文
        listener_context = context.get("listener_context", "")
        
        # 构建包含五个步骤的提示
        prompt = f"""作为回应者，我需要通过情感思维链(ECoT)来生成一个具有{emotion_condition}的回应。

查询: {query}

上下文:
听众: {listener_context}

请按照以下五个步骤生成响应:

步骤1 [理解上下文]: 描述对话的上下文。
步骤2 [识别他人情绪]: 确定听众的情绪并解释原因。
步骤3 [识别自我情绪]: 识别说话者(我)的情绪并解释原因。
步骤4 [管理自我情绪]: 考虑如何以同理心回应。
步骤5 [影响他人情绪]: 考虑回应对听众的影响。

最后，基于以上步骤，生成一个简短的情感响应。

回答时请使用以下JSON格式:
{{
  "step1_context": "对话上下文的详细描述",
  "step2_others_emotions": "听众情绪及原因分析",
  "step3_self_emotions": "说话者情绪及原因分析",
  "step4_managing_emotions": "如何以同理心方式回应的思考",
  "step5_influencing_emotions": "考虑回应对听众的情绪影响",
  "response": "最终情感响应",
  "emotion": "happy或sad或angry三者之一"
}}"""
        
        return prompt
    
    def _parse_ecot_response(self, text: str) -> Dict[str, Any]:
        """解析LLM的ECoT响应"""
        try:
            # 尝试从文本中提取JSON
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start < 0 or json_end <= 0:
                # 找不到JSON，返回简单响应
                return {
                    "step1_context": "未提供",
                    "step2_others_emotions": "未提供",
                    "step3_self_emotions": "未提供",
                    "step4_managing_emotions": "未提供", 
                    "step5_influencing_emotions": "未提供",
                    "response": text,
                    "emotion": "neutral"
                }
                
            json_str = text[json_start:json_end]
            parsed_result = json.loads(json_str)
            
            # 验证关键字段
            required_fields = [
                "step1_context", "step2_others_emotions", "step3_self_emotions", 
                "step4_managing_emotions", "step5_influencing_emotions", 
                "response", "emotion"
            ]
            
            for field in required_fields:
                if field not in parsed_result:
                    parsed_result[field] = "未提供"
            
            # 确保情绪是三种之一
            if parsed_result["emotion"] not in ["happy", "sad", "angry"]:
                # 默认使用neutral，映射到angry显示
                parsed_result["emotion"] = "angry"
                
            return parsed_result
            
        except Exception as e:
            logger.error(f"解析JSON响应时出错: {str(e)}")
            # 返回默认结果
            return {
                "step1_context": "解析失败",
                "step2_others_emotions": "解析失败",
                "step3_self_emotions": "解析失败",
                "step4_managing_emotions": "解析失败", 
                "step5_influencing_emotions": "解析失败",
                "response": "无法解析响应",
                "emotion": "neutral",
                "error": str(e)
            }
    
    def _mock_empathetic_response(self, query: str, context: Dict[str, Any], emotion_condition: str) -> Dict[str, Any]:
        """当LLM处理器不可用时模拟ECoT响应"""
        
        # 提取听众信息
        listener_context = context.get("listener_context", "")
        
        # 默认回应
        response = "我理解你的感受，这确实是个难题。相信你的判断，优先考虑自己的健康。"
        emotion = "happy"
        
        # 基于上下文简单调整回应
        if "焦虑" in listener_context:
            step2 = "听众表达了对辞职的焦虑，虽然当前工作压力大，但薪水不错，这导致了决策困难。"
            response = "我能感受到你的焦虑，在高薪和健康间做选择确实很难。相信你的判断，你的健康同样重要。"
            emotion = "sad"
        elif "压力" in listener_context:
            step2 = "听众感到工作压力很大，虽然薪水不错，但压力正在影响生活质量。"
            response = "工作压力确实会影响生活质量，即使薪水不错。照顾好自己，做对你最好的选择。"
            emotion = "angry"
        else:
            step2 = "听众正面临职业决策的困境，表现出犹豫和不确定性。"
        
        # 构建模拟的ECoT响应
        return {
            "step1_context": f"听众正在考虑是否辞去当前工作。查询是'{query}'，上下文是'{listener_context}'。",
            "step2_others_emotions": step2,
            "step3_self_emotions": "作为回应者，我感到同理心和关心，希望能够支持听众做出适合自己的决定。",
            "step4_managing_emotions": "我需要表达理解和支持，同时尊重听众的决策能力，不应给出过于直接的建议。",
            "step5_influencing_emotions": "我的回应应该让听众感到被理解，减轻其决策压力，增强自信心。",
            "response": response,
            "emotion": emotion,
            "processing_time": "0.01秒"
        } 