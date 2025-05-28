import random
import time
import jieba
import re
import logging
import os
import sys

# Add the current directory to the path to ensure absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_analyzer")

# Import the prompt manager for system prompts
from utils.prompt_manager import get_emotion_prompt

# 尝试导入LLM工厂
try:
    from .llm_factory import get_llm_handler
    LLM_SUPPORTED = True
except ImportError:
    logger.warning("LLM支持未启用，仅使用规则模式")
    LLM_SUPPORTED = False

# 尝试使用绝对导入
try:
    from config.llm_config import get_llm_config
except ImportError:
    logger.warning("无法导入LLM配置，仅使用规则模式")

# 情绪关键词词典
EMOTION_KEYWORDS = {
    'happy': ['开心', '高兴', '快乐', '欢喜', '幸福', '喜悦', '欣慰', '欢乐', '愉快', '满足', 
              '开朗', '乐观', '微笑', '大笑', '轻松', '爽', '赞', '爱', '喜欢', '好', '棒'],
    
    'angry': ['生气', '愤怒', '恼火', '发怒', '气愤', '不满', '不爽', '讨厌', '烦', '恨', '恼怒', 
              '厌恶', '憎恨', '可恶', '痛恨', '抓狂', '烦躁', '暴怒', '怨恨', '怨气'],
    
    'sad': ['悲伤', '难过', '伤心', '痛苦', '忧伤', '悲痛', '哀伤', '哀痛', '悲哀', '忧郁', 
            '伤感', '痛心', '心痛', '哭', '泪', '叹息', '沮丧', '失落', '失望', '消沉'],
    
    'surprised': ['惊讶', '吃惊', '震惊', '惊奇', '惊喜', '意外', '不可思议', '难以置信', '出乎意料', 
                 '奇怪', '惊呆', '惊诧', '惊愕', '惊叹', '惊慌', '惊恐', '惊吓', '诧异', '啊', '呀'],
    
    'calm': ['平静', '平和', '安详', '沉稳', '镇定', '冷静', '安宁', '淡定', '恬静', '安详', 
            '静谧', '安然', '坦然', '从容', '安定', '安心', '舒心', '踏实', '自在', '宁静']
}

def analyze_text(text):
    """
    分析文本情绪
    
    参数:
        text (str): 输入文本
        
    返回:
        dict: 情绪分析结果
    """
    start_time = time.time()
    
    # 检查是否使用LLM进行分析
    try:
        config = get_llm_config()
        use_llm = config.get("mode") != "rule" and LLM_SUPPORTED
    except Exception as e:
        logger.warning(f"获取LLM配置失败: {str(e)}，使用规则模式")
        use_llm = False
    
    if use_llm:
        logger.info(f"使用LLM分析文本情绪")
        
        try:
            # 获取LLM处理器
            llm_handler = get_llm_handler()
            
            # 获取系统提示
            system_prompt = get_emotion_prompt('text')
            
            # 使用LLM分析情绪
            result = llm_handler.analyze_text_with_llm(text, system_prompt)
            
            # 将LLM返回的情绪映射到猫头鹰情绪标签
            if result.get('emotion'):
                original_emotion = result.get('emotion')
                if original_emotion in ['happy', 'surprised']:
                    result['emotion'] = 'happy'  # 开心
                elif original_emotion == 'sad':
                    result['emotion'] = 'sad'    # 悲伤
                else:  # angry 或 calm
                    result['emotion'] = 'angry'  # 生气/无聊
            
            # 添加处理时间
            end_time = time.time()
            process_time = end_time - start_time
            result['processing_time'] = f"{process_time:.2f}秒"
            
            # 添加文本统计信息
            try:
                words = list(jieba.cut(text))
                result['text_stats'] = {
                    'word_count': len(words),
                    'char_count': len(text),
                }
            except Exception as e:
                logger.warning(f"计算文本统计信息时出错: {str(e)}")
            
            # 添加来源标签
            result['source'] = llm_handler.mode
            
            # 删除情感分数字段，不再使用
            if 'sentiment_score' in result:
                del result['sentiment_score']
            
            return result
            
        except Exception as e:
            logger.error(f"LLM分析出错，回退到规则模式: {str(e)}")
            # 继续使用规则模式作为备选
    
    # 规则模式（原始流程）
    logger.info("使用规则模式分析文本情绪")
    
    # 文本预处理
    text = preprocess_text(text)
    
    # 分词
    words = list(jieba.cut(text))
    
    # 基于关键词的情绪分析
    emotion_scores = calculate_emotion_scores(words)
    
    # 获取主要情绪
    emotion, _ = get_primary_emotion(emotion_scores)
    
    # 提取关键词
    keywords = extract_keywords(words, emotion)
    
    # 生成摘要
    summary = generate_summary(text, emotion, keywords)
    
    # 计算处理时间
    end_time = time.time()
    process_time = end_time - start_time
    
    # 返回结果
    return {
        'emotion': emotion,
        'keywords': keywords,
        'summary': summary,
        'processing_time': f"{process_time:.2f}秒",
        'text_stats': {
            'word_count': len(words),
            'char_count': len(text),
            'emotion_distribution': emotion_scores
        },
        'source': 'rule'  # 添加来源标识
    }

def preprocess_text(text):
    """文本预处理，去除特殊字符和多余空格"""
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', text)  # 保留中文、字母、数字和空格
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    return text

def calculate_emotion_scores(words):
    """计算各种情绪的得分"""
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS.keys()}
    total_words = len(words)
    
    if total_words == 0:
        return {emotion: random.uniform(0, 0.3) for emotion in scores.keys()}
    
    # 计算每种情绪的关键词出现次数
    for word in words:
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if word in keywords:
                scores[emotion] += 1
    
    # 添加随机因素，避免所有情绪得分为0
    has_scores = any(score > 0 for score in scores.values())
    if not has_scores:
        # 如果没有匹配任何关键词，随机选择一种情绪
        random_emotion = random.choice(list(scores.keys()))
        scores[random_emotion] = random.uniform(0.3, 0.7) * total_words
    
    # 归一化得分
    total_score = sum(scores.values()) or 1
    for emotion in scores:
        base_score = scores[emotion] / total_score
        # 添加一些随机性
        random_factor = random.uniform(0.8, 1.2)
        scores[emotion] = min(1.0, base_score * random_factor)
    
    return scores

def get_primary_emotion(emotion_scores):
    """获取主要情绪标签，根据猫头鹰表情简化为三种情感分类"""
    primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    
    # 将五种情绪简化为三种猫头鹰情绪标签
    if primary_emotion[0] in ['happy', 'surprised']:
        owl_emotion = 'happy'  # 开心
    elif primary_emotion[0] == 'sad':
        owl_emotion = 'sad'    # 悲伤
    else:  # angry 或 calm
        owl_emotion = 'angry'  # 生气/无聊 (calm也归类为无聊)
    
    return owl_emotion, 0  # 返回0作为占位符，但不再使用情感分数

def extract_keywords(words, emotion):
    """提取文本中的关键词"""
    # 基于词频和情绪相关性提取关键词
    word_freq = {}
    for word in words:
        if len(word) > 1:  # 忽略单字词
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按词频排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 选择前几个词作为关键词
    top_keywords = [word for word, _ in sorted_words[:5]]
    
    # 如果关键词不足，从对应情绪关键词中随机添加
    if len(top_keywords) < 3:
        additional_keywords = random.sample(EMOTION_KEYWORDS[emotion], min(3, len(EMOTION_KEYWORDS[emotion])))
        top_keywords.extend(additional_keywords)
        top_keywords = list(set(top_keywords))[:5]  # 去重并限制数量
    
    return top_keywords[:min(5, len(top_keywords))]

def generate_summary(text, emotion, keywords):
    """生成文本摘要"""
    # 简单摘要模板
    summary_templates = {
        'happy': [
            "文本表达了{kw}的情绪，整体氛围积极向上。",
            "内容充满了{kw}的元素，呈现出愉快的情绪状态。",
            "从文字中可以感受到{kw}的氛围，表达了积极乐观的态度。"
        ],
        'angry': [
            "文本流露出{kw}的情绪，整体语调较为激烈。",
            "内容中表现出{kw}的特征，呈现出不满和愤怒的情绪。",
            "从表达方式可以感受到{kw}的状态，情绪倾向于生气和不满。"
        ],
        'sad': [
            "文本透露出{kw}的情绪，整体氛围偏向低落。",
            "内容中流露出{kw}的特点，呈现悲伤和忧郁的状态。",
            "从文字中能够感受到{kw}的氛围，表达了失落和伤感。"
        ],
        'surprised': [
            "文本表现出{kw}的情绪，整体氛围充满惊讶。",
            "内容中流露出{kw}的特点，呈现出意外和惊奇的反应。",
            "从表达方式可以感受到{kw}的状态，表现出惊讶和诧异。"
        ],
        'calm': [
            "文本呈现出{kw}的特点，整体氛围平和安宁。",
            "内容表达了{kw}的状态，呈现出沉稳和冷静的特质。",
            "从文字中可以感受到{kw}的氛围，展现了平静和镇定的态度。"
        ]
    }
    
    # 为简短文本设置特殊模板
    if len(text) < 10:
        templates = ["简短文本「{text}」表达了{kw}的情绪。"]
    else:
        templates = summary_templates.get(emotion, ["文本中表现出{kw}的情绪。"])
    
    template = random.choice(templates)
    
    # 选择关键词填入模板
    selected_keywords = random.sample(keywords, min(len(keywords), random.randint(1, 2)))
    kw_text = "和".join(selected_keywords)
    
    if '{text}' in template:
        return template.format(text=text, kw=kw_text)
    else:
        return template.format(kw=kw_text)

# 测试函数
if __name__ == "__main__":
    test_texts = [
        "今天天气真好，我很开心！",
        "这个问题太让人生气了，我受不了了。",
        "感觉好难过，什么都不想做。",
        "哇，没想到会发生这种事，太意外了！",
        "一切都很平静，我很满足现在的生活。"
    ]
    
    for text in test_texts:
        result = analyze_text(text)
        print(f"文本: {text}")
        print(f"情绪: {result['emotion']}")
        print(f"关键词: {result['keywords']}")
        print(f"摘要: {result['summary']}")
        print(f"来源: {result.get('source', '未知')}")
        print("-" * 50) 