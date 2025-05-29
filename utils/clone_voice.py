import json
import requests
import os

# API配置
group_id = "1913402932208866274"
api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w"

# 使用相对路径获取音频文件路径
def get_voice_file_path():
    """获取语音样本文件的路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(current_dir)
    # 语音文件路径（相对于项目根目录）
    voice_file_path = os.path.join(project_root, "i.wav")
    
    return voice_file_path

def upload_voice_file():
    """上传语音文件用于克隆"""
    try:
        url = f'https://api.minimax.chat/v1/files/upload?GroupId={group_id}'
        headers = {
            'authority': 'api.minimax.chat',
            'Authorization': f'Bearer {api_key}'
        }
        
        data = {
            'purpose': 'voice_clone'
        }
        
        # 获取语音文件路径
        voice_file_path = get_voice_file_path()
        
        if not os.path.exists(voice_file_path):
            print(f"错误：文件不存在 - {voice_file_path}")
            return None
            
        with open(voice_file_path, 'rb') as f:
            files = {
                'file': f
            }
            print("正在上传语音文件...")
            response = requests.post(url, headers=headers, data=data, files=files)
            
        if response.status_code != 200:
            print(f"上传失败：HTTP {response.status_code}")
            print(f"错误信息：{response.text}")
            return None
            
        result = response.json()
        if "file" not in result:
            print("错误：响应中没有file字段")
            print(f"完整响应：{result}")
            return None
            
        file_id = result["file"].get("file_id")
        if not file_id:
            print("错误：未获取到file_id")
            return None
            
        print(f"文件上传成功，file_id: {file_id}")
        return file_id
        
    except Exception as e:
        print(f"上传过程中发生错误：{str(e)}")
        return None

def clone_voice(file_id):
    """执行语音克隆"""
    try:
        url = f'https://api.minimax.chat/v1/voice_clone?GroupId={group_id}'
        payload = json.dumps({
            "file_id": file_id,
            "voice_id": "ppooiudiiii"  # 使用基础语音ID
        })
        headers = {
            'Authorization': f'Bearer {api_key}',
            'content-type': 'application/json'
        }
        
        print("正在执行语音克隆...")
        response = requests.request("POST", url, headers=headers, data=payload)
        
        if response.status_code != 200:
            print(f"克隆失败：HTTP {response.status_code}")
            print(f"错误信息：{response.text}")
            return None
            
        result = response.json()
        print("克隆响应：", result)
        return result
        
    except Exception as e:
        print(f"克隆过程中发生错误：{str(e)}")
        return None

def main():
    """主函数"""
    print("开始语音克隆流程...")
    
    # 上传文件
    file_id = upload_voice_file()
    if not file_id:
        print("文件上传失败，终止流程")
        return
        
    # 执行克隆
    result = clone_voice(file_id)
    if not result:
        print("语音克隆失败")
        return
        
    print("语音克隆流程完成")

if __name__ == "__main__":
    main() 