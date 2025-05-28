import random
import os
import time
from PIL import Image
import logging
import sys

# Add the current directory to the path to ensure absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion_generator")

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

def generate_random_emotion():
    """
    生成随机情绪
    
    返回:
        str: 随机选择的情绪类型
    """
    emotions = ['happy', 'angry', 'sad', 'surprised', 'calm']
    return random.choice(emotions)

def generate_sentiment_score():
    """
    生成情感分数
    
    返回:
        float: -1.0到1.0之间的随机分数
    """
    return round(random.uniform(-1.0, 1.0), 2)

def generate_keywords(emotion):
    """
    基于情绪生成相关的关键词
    
    参数:
        emotion (str): 情绪类型
        
    返回:
        list: 关键词列表
    """
    keywords_map = {
        'happy': ['微笑', '愉快', '欢乐', '幸福', '积极', '阳光'],
        'angry': ['生气', '愤怒', '不满', '不悦', '冲突', '激动'],
        'sad': ['悲伤', '失落', '遗憾', '无奈', '忧郁', '消沉'],
        'surprised': ['惊讶', '意外', '震惊', '难以置信', '好奇', '疑惑'],
        'calm': ['平静', '安详', '放松', '冷静', '中性', '沉稳']
    }
    
    # 从对应情绪的关键词中随机选择3-5个
    available_keywords = keywords_map.get(emotion, ['未知'])
    count = min(len(available_keywords), random.randint(3, 5))
    
    return random.sample(available_keywords, count)

def generate_summary(emotion, keywords):
    """
    基于情绪和关键词生成简短总结
    
    参数:
        emotion (str): 情绪类型
        keywords (list): 关键词列表
        
    返回:
        str: 生成的总结
    """
    summary_templates = {
        'happy': [
            "图片中显示了{kw}的氛围，整体表现出愉快的情绪。",
            "从分析结果看，这是一张充满{kw}的图片，呈现积极向上的状态。",
            "图像内容展现了{kw}的特点，情绪倾向于开心愉悦。"
        ],
        'angry': [
            "图片中表现出{kw}的情绪，整体氛围较为紧张。",
            "分析显示，图像内容带有{kw}的特征，情绪倾向于生气或不满。",
            "从图像中可以感受到{kw}的状态，呈现出明显的愤怒情绪。"
        ],
        'sad': [
            "图片传达了{kw}的氛围，整体情绪偏向低落。",
            "分析结果表明，图像中包含{kw}的元素，呈现出悲伤的情绪状态。",
            "从图片中可以感受到{kw}的情绪，整体氛围较为消沉。"
        ],
        'surprised': [
            "图片中展现了{kw}的状态，整体呈现惊讶的情绪。",
            "分析表明，图像中的内容带有{kw}的特点，表现出意外或惊讶的反应。",
            "从图片中可以看出{kw}的表现，情绪倾向于惊讶和好奇。"
        ],
        'calm': [
            "图片呈现出{kw}的氛围，整体情绪平和稳定。",
            "分析结果显示，图像内容具有{kw}的特征，呈现出平静的状态。",
            "从图片中可以感受到{kw}的氛围，表现出冷静和沉稳的情绪。"
        ]
    }
    
    # 从当前情绪对应的模板中随机选择一个
    templates = summary_templates.get(emotion, ["图片情绪分析结果显示为{kw}。"])
    template = random.choice(templates)
    
    # 随机选择1-2个关键词填入模板
    selected_keywords = random.sample(keywords, min(len(keywords), random.randint(1, 2)))
    kw_text = "和".join(selected_keywords)
    
    return template.format(kw=kw_text)

def analyze_image_basic(image_path):
    """
    对图像进行简单的分析
    
    参数:
        image_path (str): 图像文件路径
        
    返回:
        dict: 包含图像的基本属性
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        format = img.format
        mode = img.mode
        
        # 获取文件大小
        file_size = os.path.getsize(image_path)
        file_size_kb = file_size / 1024
        
        # 主色提取（简化版）
        try:
            # 缩小图像以加快处理速度
            img_small = img.resize((100, 100))
            # 转换为RGB（如果不是）
            if img_small.mode != 'RGB':
                img_small = img_small.convert('RGB')
                
            # 获取像素数据
            pixels = list(img_small.getdata())
            
            # 计算平均颜色（简单方法）
            r_total = sum(p[0] for p in pixels)
            g_total = sum(p[1] for p in pixels)
            b_total = sum(p[2] for p in pixels)
            
            pixel_count = len(pixels)
            avg_color = (
                r_total // pixel_count,
                g_total // pixel_count,
                b_total // pixel_count
            )
            
            # 将RGB转换为十六进制
            avg_color_hex = "#{:02x}{:02x}{:02x}".format(*avg_color)
            
        except Exception as e:
            logger.warning(f"提取图像颜色时出错: {str(e)}")
            avg_color_hex = "#000000"
        
        return {
            'dimensions': f"{width}x{height}",
            'format': format,
            'mode': mode,
            'size': f"{file_size_kb:.2f} KB",
            'file_path': image_path,  # 添加文件路径
            'avg_color': avg_color_hex  # 添加平均颜色
        }
    except Exception as e:
        logger.error(f"图像分析失败: {str(e)}")
        return {
            'error': str(e),
            'file_path': image_path
        }

def generate_emotion(image_path):
    """
    生成完整的情绪分析结果
    
    参数:
        image_path (str): 图像文件路径
        
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
        logger.info(f"使用LLM分析图片情绪: {image_path}")
        
        # 获取图像基本信息
        image_info = analyze_image_basic(image_path)
        image_info['image_path'] = image_path  # 确保路径信息存在
        
        try:
            # 获取LLM处理器
            llm_handler = get_llm_handler()
            
            # 获取系统提示 - Use prompt_manager
            system_prompt = get_emotion_prompt('image')
            
            # 使用LLM分析情绪
            result = llm_handler.analyze_emotion_with_llm(image_info, system_prompt)
            
            # 添加处理时间
            end_time = time.time()
            process_time = end_time - start_time
            result['processing_time'] = f"{process_time:.2f}秒"
            
            # 添加图像信息
            result['image_info'] = image_info
            
            return result
            
        except Exception as e:
            logger.error(f"LLM分析出错，回退到规则模式: {str(e)}")
            # 继续使用规则模式作为备选
    
    # 规则模式（原始代码路径）
    logger.info(f"使用规则模式分析图片情绪: {image_path}")
    
    # 模拟处理时间（原始代码保留但不再需要）
    # processing_time = random.uniform(0.5, 2.0)
    # time.sleep(processing_time)
    
    # 生成主要情绪
    emotion = generate_random_emotion()
    
    # 生成情感分数
    sentiment_score = generate_sentiment_score()
    
    # 生成关键词
    keywords = generate_keywords(emotion)
    
    # 生成总结
    summary = generate_summary(emotion, keywords)
    
    # 计算实际处理时间
    end_time = time.time()
    process_time = end_time - start_time
    
    # 返回完整结果
    return {
        'emotion': emotion,
        'sentiment_score': sentiment_score,
        'keywords': keywords,
        'summary': summary,
        'processing_time': f"{process_time:.2f}秒",
        'image_info': analyze_image_basic(image_path),
        'source': 'rule'  # 添加来源标识
    } 