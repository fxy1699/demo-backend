"""
Script to set up Ollama with required models
"""

import os
import sys
import subprocess
import logging
import time
from utils.ollama_handler import OllamaHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup_ollama")

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama"""
    logger.info("Installing Ollama...")
    
    # Download Ollama installer
    try:
        # Create downloads directory if it doesn't exist
        os.makedirs("downloads", exist_ok=True)
        
        # Download installer
        subprocess.run([
            'curl', '-L',
            'https://ollama.com/download/OllamaSetup.exe',
            '-o', 'downloads/OllamaSetup.exe'
        ], check=True)
        
        # Run installer
        subprocess.run(['downloads/OllamaSetup.exe'], check=True)
        
        logger.info("Ollama installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Ollama: {str(e)}")
        return False

def wait_for_ollama_service(timeout=60):
    """Wait for Ollama service to be ready"""
    logger.info("Waiting for Ollama service...")
    
    handler = OllamaHandler()
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if handler.is_available():
            logger.info("Ollama service is ready")
            return True
        time.sleep(1)
    
    logger.error("Timeout waiting for Ollama service")
    return False

def pull_models():
    """Pull required models"""
    handler = OllamaHandler()
    
    # Required models
    models = [
        "qwen2.5:7b",  # Base model for text generation
        "llava:7b"     # Multimodal model for image+text
    ]
    
    for model in models:
        logger.info(f"Pulling model: {model}")
        if handler.pull_model(model):
            logger.info(f"Successfully pulled {model}")
        else:
            logger.error(f"Failed to pull {model}")

def main():
    """Main setup function"""
    logger.info("Starting Ollama setup...")
    
    # Check if Ollama is already installed
    if check_ollama_installed():
        logger.info("Ollama is already installed")
    else:
        logger.info("Ollama not found, installing...")
        if not install_ollama():
            logger.error("Failed to install Ollama")
            return False
    
    # Wait for service to be ready
    if not wait_for_ollama_service():
        logger.error("Ollama service not available")
        return False
    
    # Pull required models
    pull_models()
    
    logger.info("Ollama setup completed")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 