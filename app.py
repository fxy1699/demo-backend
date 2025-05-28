import os
import random
from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import io
import base64
from utils.emotion_generator import generate_emotion
from utils.text_analyzer import analyze_text
from utils.prompt_manager import EMOTION_ANALYSIS_PROMPTS, get_system_prompt, get_emotion_prompt
import logging
import asyncio
import sys
import traceback
from collections import deque
from datetime import datetime
import ssl
import time
from utils.ecot_handler import ECoTHandler

# Add the current directory to the path to ensure absolute imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules with absolute paths
from config.llm_config import get_llm_config, update_llm_config

# Modified import statement, use MinimaxTTS module to replace Audio module
from utils.MinimaxTTS import (
    apply_tremolo, pitch_shift, apply_pitch_envelope, blend_with_parrot_chirp,
    tts_real_parrot_play, text_to_speech_base64, text_to_speech_bytes,
    adaptive_voice_clarity, join_audio, set_voice_file_path, text_to_speech_stream
)

# Set the path for the voice file to clone - using relative path to i.m4a file
VOICE_SAMPLE_PATH = os.path.join('utils', 'audio_files', 'i.m4a')
# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Build the complete path
full_sample_path = os.path.join(current_dir, VOICE_SAMPLE_PATH)

if os.path.exists(full_sample_path):
    set_voice_file_path(full_sample_path)
    print(f"Using voice sample file for cloning: {VOICE_SAMPLE_PATH}")
else:
    print(f"Warning: Voice sample file does not exist: {full_sample_path}")
    # Try to use default voice sample
    default_path = os.path.join(current_dir, 'utils', 'audio_files', 'new_parrot_chirp.wav')
    if os.path.exists(default_path):
        set_voice_file_path(default_path)
        print(f"Using default voice sample file: new_parrot_chirp.wav")

from pydub import AudioSegment, effects

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

# Try to import LLM processor
try:
    from utils.llm_factory import get_llm_handler
    LLM_SUPPORTED = True
    # 初始化LLM处理器
    llm_handler = None
    try:
        llm_handler = get_llm_handler()
        logger.info("LLM handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM handler: {str(e)}")
        LLM_SUPPORTED = False
except ImportError as e:
    logger.warning(f"LLM support not enabled: {str(e)}")
    LLM_SUPPORTED = False
    llm_handler = None

# Initialize Flask application
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'webm', 'avi', 'mov'}

# After the imports, add a new section for context memory

# Context memory for storing recent interactions with multiple modalities
# Create a dictionary to store session-specific context histories
session_contexts = {}

# Maximum number of interactions to store in context memory
MAX_CONTEXT_SIZE = 10

# Context memory class with methods to add and retrieve context
class ContextMemory:
    def __init__(self, session_id):
        self.session_id = session_id
        self.interactions = deque(maxlen=MAX_CONTEXT_SIZE)
        self.last_access_time = datetime.now()
        
    def add_interaction(self, interaction):
        """Add a new interaction to the context memory"""
        self.interactions.append(interaction)
        self.last_access_time = datetime.now()
        
    def get_context_string(self):
        """Get formatted context string from stored interactions"""
        if not self.interactions:
            return ""
            
        context_parts = []
        for interaction in self.interactions:
            # Format based on interaction type
            if 'type' not in interaction:
                continue
                
            if interaction['type'] == 'text_only':
                context_parts.append(f"User: {interaction.get('user_input', '')}")
                context_parts.append(f"Assistant: {interaction.get('response', '')}")
            elif interaction['type'] == 'audio':
                context_parts.append(f"User (voice): {interaction.get('transcript', '')}")
                context_parts.append(f"Assistant: {interaction.get('response', '')}")
            elif interaction['type'] == 'image':
                context_parts.append(f"User: [shared an image]")
                context_parts.append(f"Assistant: {interaction.get('response', '')}")
            elif interaction['type'] == 'multimodal':
                context_parts.append(f"User: {interaction.get('transcript', '')} [shared an image]")
                context_parts.append(f"Assistant: {interaction.get('response', '')}")
        
        return "\n".join(context_parts)
        
    def update_access_time(self):
        """Update the last access time of this context"""
        self.last_access_time = datetime.now()

def get_context_memory(session_id):
    """Get or create a context memory for the given session ID"""
    if session_id not in session_contexts:
        session_contexts[session_id] = ContextMemory(session_id)
    else:
        # Update access time
        session_contexts[session_id].update_access_time()
    return session_contexts[session_id]

def cleanup_old_contexts():
    """Remove context memories that haven't been accessed for some time"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, context in session_contexts.items():
        # If not accessed in the last hour, mark for removal
        time_diff = (current_time - context.last_access_time).total_seconds()
        if time_diff > 3600:  # 1 hour
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del session_contexts[session_id]

def allowed_image_file(filename):
    """Check if image file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_audio_file(filename):
    """Check if audio file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def allowed_video_file(filename):
    """Check if video file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# 全局LLM处理器
llm_handler = None
ecot_handler = None  # 添加ECoT处理器全局变量

# 初始化
def init_llm():
    """初始化LLM处理器"""
    global llm_handler, ecot_handler
    if LLM_SUPPORTED:
        try:
            from utils.llm_handler import LLMHandler
            llm_handler = LLMHandler()
            # 初始化ECoT处理器
            ecot_handler = ECoTHandler(llm_handler)
            return True
        except Exception as e:
            logger.error(f"初始化LLM处理器失败: {str(e)}")
    return False

@app.route('/api/generate-multimodal', methods=['POST'])
def generate_multimodal():
    """
    Multimodal generation endpoint that handles text, image and voice inputs.
    
    Request Form:
    - prompt (str): User text prompt or voice transcript
    - with_audio (str): Whether to generate audio response "true" or "false"
    - audio_style (str): Audio style for voice generation
    - analyze (str): Whether to analyze the emotion "true" or "false"
    - iq_level (str, optional): IQ level for the response generation
    - session_id (str, optional): Client session ID for context memory
    - object_query (str, optional): Flag to indicate this is an object recognition query
    - ecot_enabled (str, optional): Whether to use Emotional Chain of Thought "true" or "false"
    
    Request Files:
    - image_file (file, optional): Image file to analyze
    
    Returns:
        JSON response with generated text and optional audio
    """
    try:
        # Get form data
        prompt = request.form.get('prompt', '')
        with_audio = request.form.get('with_audio', 'false') == 'true'
        audio_style = request.form.get('audio_style', 'default')
        analyze_flag = request.form.get('analyze', 'false') == 'true'
        
        # 新增：是否启用ECoT（情感思维链）
        ecot_enabled = request.form.get('ecot_enabled', 'false') == 'true'
        
        # Get IQ level for response generation
        iq_level = request.form.get('iq_level', 'grownup')
        
        # Get session ID for context memory
        session_id = request.form.get('session_id', 'default')
        
        # Check if this is an object recognition query
        is_object_query = request.form.get('object_query', 'false') == 'true'
        
        # Enhanced logging, showing all key parameters
        logger.info(f"Received multimodal generation request, prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        logger.info(f"Request parameters: analyze_flag={analyze_flag}, with_audio={with_audio}, audio_style={audio_style}, iq_level={iq_level}, ecot_enabled={ecot_enabled}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Is object recognition query: {is_object_query}")
        
        # Log all form keys in the request
        logger.info(f"Request contains the following form fields: {list(request.form.keys())}")
        
        # Print raw form data contents for debugging
        print(f"[DEBUG] Request form data received: {dict(request.form)}")
        print(f"[DEBUG] Received IQ level in backend: '{iq_level}'")
        print(f"[DEBUG] Session ID: '{session_id}'")
        print(f"[DEBUG] Is object recognition query: {is_object_query}")
        print(f"[DEBUG] ECoT enabled: {ecot_enabled}")
        
        if analyze_flag:
            logger.info("Request includes emotion analysis flag")
        
        # Get context memory for this session
        context_memory = get_context_memory(session_id)
        
        # Periodically clean up old contexts
        cleanup_old_contexts()
        
        # Get LLM handler from factory
        from utils.llm_factory import get_llm_handler
        llm_handler = get_llm_handler()
        
        # 检查ECoT处理器是否初始化
        global ecot_handler
        if ecot_enabled and ecot_handler is None:
            if not init_llm():
                logger.warning("ECoT处理器初始化失败，将使用常规处理流程")
                ecot_enabled = False
        
        # Check if image file is provided
        has_image = 'image_file' in request.files and request.files['image_file'].filename != ''
        
        # Setup system prompt based on the input type and query intent
        if has_image:
            if is_object_query:
                # Use specific object recognition prompt from backend prompt manager
                system_prompt = get_system_prompt('multimodal')
                logger.info("Using specialized object recognition prompt for image analysis")
            else:
                # For voice input with image, use multimodal prompt
                system_prompt = get_system_prompt('multimodal')
                logger.info("Using multimodal prompt for voice input with image")
        else:
            # Text-only input
            system_prompt = get_system_prompt('default')
            logger.info("Using default text-only prompt")
        
        # 如果启用ECoT，使用ECoT系统提示替换原始提示
        if ecot_enabled:
            system_prompt = get_emotion_prompt('ecot')
            logger.info("Using ECoT system prompt for emotional reasoning")
        
        # Get context from memory and add to system prompt if available
        context_string = context_memory.get_context_string()
        if context_string:
            system_prompt += f"\n\nPrevious conversation context:\n{context_string}"
            
        print(f"[DEBUG] Using system prompt: '{system_prompt[:200]}...'")
        
        # Try to save files to memory
        image_file_data = None
        
        if has_image:
            image_file = request.files['image_file']
            image_filename = secure_filename(image_file.filename)
            logger.info(f"Received image file: {image_filename}")
            # Read image into memory
            image_file_data = image_file.read()
            
        print(f"[DEBUG] Final prompt: '{prompt[:100]}...'")
        
        # Process with LLM
        text_response = "No response generated"
        result = {}
        
        # 获取LLM处理器配置信息
        llm_config = get_llm_config()
        llm_mode = llm_config.get('mode', 'rule')
        
        # 优先使用requests方式调用，减少对OpenAI客户端的依赖
        use_requests_method = True
        
        # 根据LLM处理器配置的mode来决定是否使用requests方式
        if llm_mode == 'local':
            # 本地模型必须使用Transformers库
            use_requests_method = False
        
        # 检查LLM处理器是否正确初始化
        if llm_handler is None:
            logger.error("LLM处理器未初始化，使用备用响应")
            text_response = "LLM处理器未初始化，无法生成响应。请检查配置和依赖项。"
            result = {"text": text_response, "error": "LLM handler not initialized"}
        else:
            try:
                # 如果启用ECoT模式，将使用ECoT处理器生成响应
                if ecot_enabled:
                    # 构建上下文信息
                    ecot_context = {
                        'listener_context': context_string,
                        'speaker_context': ''
                    }
                    
                    # 使用ECoT处理器生成情感响应
                    ecot_result = ecot_handler.generate_empathetic_response(
                        query=prompt,
                        context=ecot_context,
                        emotion_condition='empathy',
                        system_prompt=system_prompt
                    )
                    
                    # 将ECoT结果转换为普通响应格式
                    text_response = ecot_result.get('response', 'No response generated')
                    result = {
                        'text': text_response,
                        'emotion': ecot_result.get('emotion', 'neutral'),
                        'ecot_steps': {
                            'step1_context': ecot_result.get('step1_context', ''),
                            'step2_others_emotions': ecot_result.get('step2_others_emotions', ''),
                            'step3_self_emotions': ecot_result.get('step3_self_emotions', ''),
                            'step4_managing_emotions': ecot_result.get('step4_managing_emotions', ''),
                            'step5_influencing_emotions': ecot_result.get('step5_influencing_emotions', '')
                        }
                    }
                    
                    # 添加处理时间
                    result['processing_time'] = ecot_result.get('processing_time', 'unknown')
                    
                    # 存储到上下文记忆
                    if has_image:
                        context_memory.add_interaction({
                            'type': 'multimodal',
                            'transcript': prompt,
                            'has_image': True,
                            'response': text_response
                        })
                    else:
                        context_memory.add_interaction({
                            'type': 'text_only',
                            'user_input': prompt,
                            'response': text_response
                        })
                else:
                    # 常规响应生成流程
                    # Process the request based on input type
                    if has_image:
                        print(f"[DEBUG] Processing with image input, IQ level: {iq_level}")
                        # Handle image input
                        result = llm_handler.generate_text_with_image(
                            prompt=prompt,
                            image_file_data=image_file_data,
                            system_prompt=system_prompt,
                            max_tokens=1024,
                            temperature=0.7,
                            top_p=0.9
                        )
                        text_response = result.get('text', 'No response generated')
                        # Store interaction in context memory
                        context_memory.add_interaction({
                            'type': 'multimodal',
                            'transcript': prompt,
                            'has_image': True,
                            'response': text_response
                        })
                    else:
                        # Text-only input
                        print(f"[DEBUG] Processing text-only with IQ level: {iq_level}")
                        result = llm_handler.generate_text(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            max_tokens=1024,
                            temperature=0.7,
                            top_p=0.9
                        )
                        text_response = result.get('text', 'No response generated')
                        # Store interaction in context memory
                        context_memory.add_interaction({
                            'type': 'text_only',
                            'user_input': prompt,
                            'response': text_response
                        })
                
                logger.info(f"Original response: {text_response[:100]}")
                
                # Generate audio response if requested
                audio_base64 = None
                if with_audio and text_response:
                    try:
                        # Generate audio response with the modified text
                        print(f"生成音频响应，文本: '{text_response[:30]}...'，风格: {audio_style}")
                        audio_base64 = text_to_speech_base64(
                            text=text_response, 
                            style=audio_style
                        )
                        if audio_base64:
                            logger.info(f"成功生成音频响应，大小: {len(audio_base64)} 字符，风格: {audio_style}")
                        else:
                            logger.error(f"音频生成失败，没有返回数据")
                    except Exception as audio_error:
                        logger.error(f"Error generating audio response: {str(audio_error)}")
                        traceback.print_exc()
                
                # Create response with the modified text
                response = {
                    'text': text_response,
                    'audio': audio_base64
                }
                
                # 如果是ECoT模式，添加ECoT步骤信息
                if ecot_enabled and 'ecot_steps' in result:
                    response['ecot_steps'] = result['ecot_steps']
                
                # Include emotion analysis if available
                emotion_data = {'emotion': 'neutral'}  # Default emotion
                if analyze_flag:
                    try:
                        # If the request includes the emotion analysis flag, ensure emotion analysis is performed
                        logger.info("Performing emotion analysis")
                        
                        # 如果启用ECoT，直接使用ECoT提供的情绪
                        if ecot_enabled and 'emotion' in result:
                            emotion_data = {'emotion': result['emotion']}
                            logger.info(f"Using ECoT emotion: {emotion_data['emotion']}")
                        else:
                            emotion_result = llm_handler.analyze_text_with_llm(text_response)
                            logger.info(f"Emotion analysis returned result: {emotion_result}")
                            
                            if emotion_result and 'emotion' in emotion_result:
                                emotion_data = emotion_result
                                logger.info(f"Emotion analysis result: {emotion_data['emotion']}")
                            else:
                                # If the analysis result doesn't contain the emotion field, try to infer emotion from the text content
                                logger.info("Emotion analysis didn't return a valid result, attempting to infer emotion from text content")
                                emotion_data = infer_emotion_from_text(text_response)
                                logger.info(f"Emotion inferred from text: {emotion_data['emotion']}")
                    except Exception as e:
                        logger.error(f"Error analyzing emotion: {str(e)}")
                        # Even if emotion analysis fails, retain the default emotion
                        logger.info(f"Emotion analysis failed, using default emotion: {emotion_data['emotion']}")
                
                # Ensure emotion is one of the three supported types: happy, sad, angry
                supported_emotions = ['happy', 'sad', 'angry']
                if emotion_data['emotion'] not in supported_emotions:
                    # Map other emotions to the closest supported emotion
                    if emotion_data['emotion'] in ['boring', 'neutral', 'normal']:
                        emotion_data['emotion'] = 'angry'  # Map boring and neutral to angry
                    elif emotion_data['emotion'] in ['surprised', 'excited']:
                        emotion_data['emotion'] = 'happy'  # Map surprised and excited to happy
                    else:
                        emotion_data['emotion'] = 'angry'  # Default map to angry
                    
                    logger.info(f"Mapped non-standard emotion to supported emotion type: {emotion_data['emotion']}")
                
                # Include emotion analysis in the response
                response.update(emotion_data)
                
                # Add session ID to the response
                response['session_id'] = session_id
                
                # Add a flag indicating if this was an object recognition query
                response['is_object_query'] = is_object_query
                
                # 添加标志指示是否使用了ECoT
                response['ecot_enabled'] = ecot_enabled
                
                # Print summary of the response content to be returned
                logger.info(f"Response summary: text_length={len(response.get('text', ''))} characters, emotion={response.get('emotion', 'none')}")
                if 'audio' in response:
                    logger.info("Response includes audio data")
                
                return jsonify(response)
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                traceback.print_exc()
                return jsonify({'error': f'Failed to process request: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Multimodal generation failed: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-ecot', methods=['POST'])
def generate_ecot():
    """
    生成情感思维链(ECoT)响应
    
    请求参数:
        - Content-Type: application/json
        - 请求体:
            - query: 查询文本
            - listener_context: 听众上下文
            - emotion_condition: 情绪条件(默认empathy)
    
    响应:
        - Content-Type: application/json
        - 成功: {
            'step1_context': '上下文描述',
            'step2_others_emotions': '他人情绪分析',
            'step3_self_emotions': '自我情绪分析',
            'step4_managing_emotions': '情绪管理考虑',
            'step5_influencing_emotions': '情绪影响考虑',
            'response': '情感响应',
            'emotion': '情绪标签',
            'processing_time': '处理时间'
        }
        - 失败: {'error': 'error_message'}
    """
    start_time = time.time()
    
    if not LLM_SUPPORTED:
        return jsonify({'error': 'LLM功能未启用'}), 501
    
    # 检查ecot_handler是否初始化
    global ecot_handler
    if ecot_handler is None:
        if not init_llm():
            return jsonify({'error': 'ECoT处理器未初始化'}), 500
    
    try:
        # 获取请求参数
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        query = data.get('query', '')
        if not query:
            return jsonify({'error': '查询文本不能为空'}), 400
        
        # 构建上下文
        context = {
            'listener_context': data.get('listener_context', ''),
            'speaker_context': data.get('speaker_context', '')
        }
        
        # 获取情绪条件
        emotion_condition = data.get('emotion_condition', 'empathy')
        
        # 调用ECoT处理器生成响应
        result = ecot_handler.generate_empathetic_response(
            query=query,
            context=context,
            emotion_condition=emotion_condition,
            system_prompt=get_emotion_prompt('ecot')
        )
        
        # 计算总处理时间
        end_time = time.time()
        process_time = end_time - start_time
        result['total_processing_time'] = f"{process_time:.2f}秒"
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"生成ECoT响应时出错: {str(e)}")
        return jsonify({'error': f'生成ECoT响应失败: {str(e)}'}), 500

@app.route('/api/llm/models', methods=['GET'])
def llm_models_api():
    """
    Get list of available LLM models
    
    Response:
        - Content-Type: application/json
        - Success: {
            'models': {
                'api': [{'id': 'model_id', 'name': 'model_name'}, ...],
                'ollama': [{'id': 'model_id', 'name': 'model_name'}, ...],
                'local': [{'id': 'model_id', 'name': 'model_name'}, ...]
            },
            'current': {
                'mode': 'current_mode',
                'model': {'id': 'model_id', 'name': 'model_name'}
            }
        }
        - Failure: {'error': error_message}
    """
    if not LLM_SUPPORTED:
        return jsonify({'error': 'LLM functionality not enabled'}), 501
        
    if llm_handler is None:
        return jsonify({'error': 'LLM handler not initialized'}), 500
        
    try:
        # Get current configuration
        config = get_llm_config()
        
        # Initialize model list
        models = {
            'api': [
                {'id': 'qwen-omni-turbo', 'name': 'Qwen Omni Turbo (Online)'},
                {'id': 'qwen-turbo', 'name': 'Qwen Turbo (Online)'},
                {'id': 'qwen-plus', 'name': 'Qwen Plus (Online)'}
            ],
            'ollama': [
                {'id': 'llava:7b', 'name': 'LLaVA 7B (Multimodal)'},
                {'id': 'qwen2.5:7b', 'name': 'Qwen 2.5 7B'},
                {'id': 'llama3:8b', 'name': 'Llama 3 8B'}
            ],
            'local': []
        }
        
        # Get current mode and model
        current_mode = config.get('mode', 'rule')
        current_model_id = ''
        current_model_name = ''
        
        # Get current model information based on mode
        if current_mode == 'api':
            current_model_id = config.get('api', {}).get('model_name', '')
            # Look for name in the API model list
            for model in models['api']:
                if model['id'] == current_model_id:
                    current_model_name = model['name']
                    break
            if not current_model_name:
                current_model_name = current_model_id
                
        elif current_mode == 'local':
            current_model_id = config.get('local', {}).get('model_path', '')
            current_model_name = os.path.basename(current_model_id) if current_model_id else 'Unknown local model'
            
        # Try to get Ollama model list
        try:
            # Here you can try to import and use OllamaHandler to get the model list
            from utils.ollama_handler import OllamaHandler
            ollama = OllamaHandler(
                base_url=config.get('ollama', {}).get('base_url', 'http://127.0.0.1:11434'),
                api_key=config.get('ollama', {}).get('api_key', '')
            )
            ollama_models = ollama.get_available_models()
            
            if ollama_models:
                # Merge default models and installed models
                installed_models = [{'id': name, 'name': name} for name in ollama_models 
                                 if not any(m['id'] == name for m in models['ollama'])]
                models['ollama'].extend(installed_models)
                
                # If current mode is ollama, set current model name
                if current_mode == 'ollama':
                    current_model_id = config.get('ollama', {}).get('models', {}).get('multimodal', {}).get('name', 'llava:7b')
                    # Look for name in the Ollama model list
                    for model in models['ollama']:
                        if model['id'] == current_model_id:
                            current_model_name = model['name']
                            break
                    if not current_model_name:
                        current_model_name = current_model_id
                
        except Exception as e:
            logger.warning(f"Failed to get Ollama model list: {str(e)}")
            # Use default Ollama model list
        
        # Return result
        return jsonify({
            'models': models,
            'current': {
                'mode': current_mode,
                'model': {
                    'id': current_model_id,
                    'name': current_model_name
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model list: {str(e)}")
        return jsonify({'error': f'Failed to get model list: {str(e)}'}), 500

@app.route('/api/llm/set-model', methods=['POST'])
def set_llm_model_api():
    """
    Set current LLM model and mode
    
    Request:
        - Content-Type: application/json
        - Parameters:
            - mode: Mode, can be 'api', 'ollama', 'local', 'rule'
            - model_id: Model ID
    
    Response:
        - Content-Type: application/json
        - Success: {
            'success': true,
            'mode': 'current_mode',
            'model': {
                'id': 'model_id',
                'name': 'model_name'
            }
        }
        - Failure: {'error': error_message}
    """
    if not LLM_SUPPORTED:
        return jsonify({'error': 'LLM functionality not enabled'}), 501
        
    if llm_handler is None:
        return jsonify({'error': 'LLM handler not initialized'}), 500
        
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No request parameters provided'}), 400
            
        mode = data.get('mode')
        model_id = data.get('model_id', '')
        
        if not mode:
            return jsonify({'error': 'No mode parameter provided'}), 400
            
        # Validate mode
        if mode not in ['api', 'ollama', 'local', 'rule']:
            return jsonify({'error': f'Unsupported mode: {mode}'}), 400
            
        # Prepare updated configuration
        new_config = {'mode': mode}
        
        # Set model configuration based on mode
        if mode == 'api' and model_id:
            new_config['api'] = {
                'model_name': model_id
            }
        elif mode == 'ollama' and model_id:
            # Ensure ollama configuration exists
            current_config = get_llm_config()
            ollama_config = current_config.get('ollama', {})
            
            # Determine if it's a multimodal model
            is_multimodal = 'llava' in model_id.lower()
            
            # Update relevant model configuration
            if is_multimodal:
                ollama_config['models'] = ollama_config.get('models', {})
                ollama_config['models']['multimodal'] = {
                    'name': model_id,
                    'context_window': 4096,
                    'system_prompt': "You are a helpful multimodal AI assistant that can understand both images and text."
                }
            else:
                ollama_config['models'] = ollama_config.get('models', {})
                ollama_config['models']['text'] = {
                    'name': model_id,
                    'context_window': 8192,
                    'system_prompt': "You are a helpful AI assistant."
                }
            
            new_config['ollama'] = ollama_config
            
        elif mode == 'local' and model_id:
            new_config['local'] = {
                'model_path': model_id
            }
        
        # Update configuration
        updated_config = update_llm_config(new_config)
        
        # Determine model name
        model_name = model_id
        if mode == 'api':
            # Common API model name mapping
            api_model_names = {
                'qwen-omni-turbo': 'Qwen Omni Turbo (Online)',
                'qwen-turbo': 'Qwen Turbo (Online)',
                'qwen-plus': 'Qwen Plus (Online)'
            }
            model_name = api_model_names.get(model_id, model_id)
        elif mode == 'local':
            model_name = os.path.basename(model_id) if model_id else 'Local model'
        
        # Return success response
        return jsonify({
            'success': True,
            'mode': mode,
            'model': {
                'id': model_id,
                'name': model_name
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting model: {str(e)}")
        return jsonify({'error': f'Failed to set model: {str(e)}'}), 500

@app.route('/api/llm/config', methods=['GET', 'POST'])
def llm_config_api():
    """
    Get or update LLM configuration
    
    Request (POST):
        - Content-Type: application/json
        - Parameters:
            - mode: Mode, can be 'api', 'ollama', 'local', 'rule'
            - api.model_name: API model name
            - api.api_key: API key
            - api.api_base_url: API base URL
            - ollama.base_url: Ollama base URL
            - ollama.models.multimodal.name: Ollama multimodal model name
            - ollama.models.text.name: Ollama text model name
    
    Response:
        - Content-Type: application/json
        - Success: {
            'config': {...current_config}
        }
        - Failure: {'error': error_message}
    """
    if not LLM_SUPPORTED:
        return jsonify({'error': 'LLM functionality not enabled'}), 501
        
    if llm_handler is None:
        return jsonify({'error': 'LLM handler not initialized'}), 500
        
    if request.method == 'GET':
        # Get current configuration
        config = get_llm_config()
        # For security, remove sensitive information
        if 'api' in config and 'api_key' in config['api']:
            config['api']['api_key'] = '******' if config['api']['api_key'] else ''
        return jsonify(config), 200
        
    elif request.method == 'POST':
        try:
            # Get new configuration
            new_config = request.get_json()
            if not new_config:
                return jsonify({'error': 'No configuration parameters provided'}), 400
                
            # Update configuration
            updated_config = update_llm_config(new_config)
            
            # For security, remove sensitive information
            if 'api' in updated_config and 'api_key' in updated_config['api']:
                updated_config['api']['api_key'] = '******' if updated_config['api']['api_key'] else ''
                
            return jsonify({
                'message': 'Configuration updated',
                'config': updated_config
            }), 200
            
        except Exception as e:
            logger.error(f"Error updating LLM configuration: {str(e)}")
            return jsonify({'error': f'Failed to update configuration: {str(e)}'}), 500

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    """Health check endpoint"""
    status_info = {
        'status': 'ok',
        'message': 'Service running normally',
        'features': {
            'llm_supported': LLM_SUPPORTED
        }
    }
    
    # Add LLM mode information
    if LLM_SUPPORTED:
        try:
            config = get_llm_config()
            current_mode = config.get('mode', 'rule')
            status_info['llm_mode'] = current_mode
            
            # Get current model information
            if current_mode == 'api':
                current_model = config.get('api', {}).get('model_name', '')
                status_info['llm_model'] = current_model
            elif current_mode == 'ollama':
                # Get currently used model (multimodal or text)
                ollama_config = config.get('ollama', {}).get('models', {})
                if 'multimodal' in ollama_config:
                    current_model = ollama_config['multimodal'].get('name', '')
                else:
                    current_model = ollama_config.get('text', {}).get('name', '')
                status_info['llm_model'] = current_model
                
                # Check Ollama service availability
                try:
                    from utils.ollama_handler import OllamaHandler
                    ollama = OllamaHandler(
                        base_url=config.get('ollama', {}).get('base_url', 'http://127.0.0.1:11434'),
                        api_key=config.get('ollama', {}).get('api_key', '')
                    )
                    status_info['ollama_available'] = ollama.is_available()
                    if status_info['ollama_available']:
                        status_info['ollama_models'] = ollama.get_available_models()
                except Exception as e:
                    status_info['ollama_available'] = False
                    status_info['ollama_error'] = str(e)
            elif current_mode == 'local':
                current_model = config.get('local', {}).get('model_path', '')
                status_info['llm_model'] = os.path.basename(current_model) if current_model else ''
            
            # Add available modes
            status_info['available_modes'] = ['api', 'ollama', 'local', 'rule']
                
        except Exception as e:
            status_info['llm_mode'] = 'error'
            status_info['error'] = str(e)
    
    return jsonify(status_info), 200

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """API endpoint to get prompts for the backend to use"""
    # Return a simplified set of prompts - these are needed for the backend
    # but are not used to override the backend prompts when processing requests
    # from utils.prompt_manager import FRONTEND_PROMPTS
    
    # Provide minimal information without exposing actual system prompts
    prompts = {
        'capabilities': {
            'image_analysis': True,
            'voice_recognition': True,
            'object_recognition': True
        }
    }
    
    return jsonify(prompts)

@app.route('/api/prompts/update', methods=['POST'])
def update_prompts():
    """Update specific frontend prompt type used by the backend"""
    try:
        # from utils.prompt_manager import FRONTEND_PROMPTS
        
        data = request.json
        # prompt_type = data.get('type')
        prompt_text = data.get('text')
        
        if not all([prompt_text]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Only allow updating frontend prompts used by the backend
        # if prompt_type in FRONTEND_PROMPTS:
        #     FRONTEND_PROMPTS[prompt_type] = prompt_text
        #     return jsonify({'success': True, 'message': f'Updated {prompt_type} prompt'})
        # else:
        #     return jsonify({'error': f'Invalid prompt type: {prompt_type}'}), 400
        else:
            return jsonify({'success': True, 'message': f'Updated {prompt_text} prompt'})
    except Exception as e:
        logger.error(f"Failed to update prompt: {str(e)}")
        return jsonify({'error': f'Failed to update prompt: {str(e)}'}), 500

@app.route('/api/prompts/voice', methods=['GET'])
def get_voice_prompt():
    """Get the voice generation prompt with IQ adjustment"""
    try:
        iq_level = request.args.get('iq', 'normal')
        
        # Basic prompt for parrot-like behavior
        base_prompt = "You are a voice assistant, please respond to users with short phrases, mimicking the way parrots speak."
        
        # Adjust based on IQ level
        if iq_level == 'potato':
            prompt = base_prompt + " Use extremely simple vocabulary, often repeat words, and sometimes say completely meaningless word combinations."
        elif iq_level == 'kiddo':
            prompt = base_prompt + " Use simple vocabulary, simple sentence structure, and occasionally say meaningless word combinations."
        elif iq_level == 'grownup':
            prompt = base_prompt + " Use standard language, complete but short sentence structure, rarely use meaningless words."
        elif iq_level == 'einstein':
            prompt = base_prompt + " Use advanced vocabulary, elegant but short sentence structure, occasionally quote classic sayings."
        else:
            prompt = base_prompt + " Use standard language, short and powerful."
            
        return jsonify({'prompt': prompt})
    except Exception as e:
        logger.error(f"Failed to get voice prompt: {str(e)}")
        return jsonify({'error': f'Failed to get voice prompt: {str(e)}'}), 500

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    # Handle preflight requests
    if request.method == 'OPTIONS':
        response.headers.add('Access-Control-Max-Age', '1728000')
    return response

# 添加新的流式语音API端点
@app.route('/api/stream-speech', methods=['POST'])
def stream_speech():
    """
    流式文本转语音API - 可以大幅减少延迟
    
    请求参数:
        - text: 要转换为语音的文本
        - style: 语音风格 (default, cartoon, slow, very_slow)
        
    返回:
        流式音频数据
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '缺少text参数'}), 400
            
        text = data['text']
        style = data.get('style', 'very_slow')  # 默认使用very_slow风格
        
        print(f"收到流式语音请求: text='{text[:30]}...'，style={style}")
        
        # 直接使用base64方法生成完整的音频数据
        audio_data = text_to_speech_bytes(text, style)
        
        if not audio_data:
            print("生成音频失败，返回空数据")
            return jsonify({'error': '音频生成失败'}), 500
            
        print(f"成功生成音频，大小: {len(audio_data)} 字节")
        
        # 返回完整的音频数据
        return Response(audio_data, mimetype='audio/mp3')
        
    except Exception as e:
        print(f"流式语音生成异常: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 修改现有的语音生成逻辑
def get_audio_response(text, style):
    """优化的音频响应生成函数，使用缓存和异步处理"""
    try:
        # 生成音频响应
        audio_base64 = text_to_speech_base64(
            text=text, 
            style=style
        )
        return audio_base64
    except Exception as audio_error:
        logger.error(f"音频生成失败: {str(audio_error)}")
        return None

# Setup Basic Authentication for public access
AUTH_USERNAME = os.environ.get('AUTH_USERNAME', 'skyrisai')
AUTH_PASSWORD = os.environ.get('AUTH_PASSWORD', 'skyrispassword')

def check_auth(username, password):
    """Check if the username and password are valid"""
    return username == AUTH_USERNAME and password == AUTH_PASSWORD

def authenticate():
    """Return a 401 response that enables basic auth"""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    """Decorator for routes that require authentication"""
    from functools import wraps
    from flask import request, Response
    import base64
    
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        remote_addr = request.remote_addr
        
        # Always allow localhost and private network access without authentication
        if remote_addr == '127.0.0.1' or remote_addr == 'localhost' or remote_addr.startswith('192.168.') or remote_addr.startswith('10.'):
            return f(*args, **kwargs)
            
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Apply the authentication decorator to all API routes 
for endpoint in app.view_functions:
    if endpoint.startswith('api'):
        app.view_functions[endpoint] = requires_auth(app.view_functions[endpoint])

# Add a function to infer emotion from text content
def infer_emotion_from_text(text):
    """Infer emotion from text content"""
    if not text:
        return {'emotion': 'neutral'}
    
    text_lower = text.lower()
    
    # Match emotional vocabulary in Chinese and English
    if any(word in text_lower for word in ['happy', 'glad', 'joy', 'excited', 'happy', 'cheerful', 'joyful', 'excited']):
        return {'emotion': 'happy'}
    elif any(word in text_lower for word in ['sad', 'sorrow', 'unhappy', 'depressed', 'sad', 'upset', 'heartbroken', 'depressed']):
        return {'emotion': 'sad'}
    elif any(word in text_lower for word in ['angry', 'mad', 'rage', 'furious', 'angry', 'irritated', 'furious', 'enraged']):
        return {'emotion': 'angry'}
    elif any(word in text_lower for word in ['surprised', 'wow', 'shock', 'amazed', 'surprised', 'shocked', 'astonished', 'amazed']):
        return {'emotion': 'surprised'}
    elif any(word in text_lower for word in ['bored', 'boring', 'tired', 'dull', 'bored', 'weary', 'tired', 'dull']):
        return {'emotion': 'angry'}  # Boredom also maps to angry expression
    else:
        return {'emotion': 'neutral'}

if __name__ == '__main__':
    # Setup for both local and public access
    local_port = 5001
    public_port = int(os.environ.get('PUBLIC_PORT', 8080))
    public_ip = "172.208.104.103"  # Set to your desired IP address
    
    # Path to SSL certificates
    ssl_cert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ssl_certs', 'cert.pem')
    ssl_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ssl_certs', 'key.pem')
    
    # Check if SSL certificates exist
    has_ssl = os.path.exists(ssl_cert_path) and os.path.exists(ssl_key_path)
    
    import threading
    from werkzeug.serving import make_server
    
    class ServerThread(threading.Thread):
        def __init__(self, app, host, port, ssl_context=None):
            threading.Thread.__init__(self)
            self.server = make_server(host, port, app, ssl_context=ssl_context)
            self.ctx = app.app_context()
            self.ctx.push()

        def run(self):
            self.server.serve_forever()

        def shutdown(self):
            self.server.shutdown()
    
    # Create SSL context
    ssl_context = None
    if has_ssl:
        ssl_context = (ssl_cert_path, ssl_key_path)
    
    # Start local server (only for localhost connections)
    local_server = ServerThread(app, 'localhost', local_port)
    local_server.daemon = True
    local_server.start()
    
    # Start public server (for any connections)
    public_server = ServerThread(app, '0.0.0.0', public_port, ssl_context=ssl_context)
    public_server.daemon = True
    public_server.start()
    
    print("\n" + "=" * 70)
    print(f"SkyrisAI Backend Server Started")
    print("=" * 70)
    print(f"\nLocal server (no auth required): http://localhost:{local_port}")
    if has_ssl:
        print(f"Public server (auth required): https://{public_ip}:{public_port}")
    else:
        print(f"Public server (auth required): http://{public_ip}:{public_port}")
        print("\nWARNING: HTTPS is not enabled! Using insecure HTTP instead.")
    print("\nPublic access credentials:")
    print(f"  Username: {AUTH_USERNAME}")
    print(f"  Password: {'*' * len(AUTH_PASSWORD)}")
    print("\nTo access from external devices, use your machine's IP address")
    if has_ssl:
        print(f"  Example: https://{public_ip}:{public_port}/api/healthcheck")
    else:
        print(f"  Example: http://{public_ip}:{public_port}/api/healthcheck")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        # Keep the main thread alive
        while True:
            threading.Event().wait(60)  # Wait for 60 seconds
    except KeyboardInterrupt:
        print("\nShutting down server...")
        local_server.shutdown()
        public_server.shutdown()
        print("Server shutdown complete.") 
