from pydub import AudioSegment
import os

def convert_m4a_to_wav(input_file, output_file=None):
    """
    将m4a文件转换为wav格式
    
    参数:
        input_file (str): 输入的m4a文件路径
        output_file (str): 输出的wav文件路径，如果不指定则使用相同文件名
    
    返回:
        str: 输出文件的路径
    """
    try:
        # 如果没有指定输出文件名，则使用输入文件名（更改扩展名）
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + '.wav'
        
        print(f"开始转换音频文件...")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        
        # 加载m4a文件
        audio = AudioSegment.from_file(input_file, format="m4a")
        
        # 导出为wav格式
        audio.export(output_file, format="wav")
        
        print(f"转换完成！")
        print(f"文件已保存到: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 指定输入文件路径
    input_file = r"D:\backend\demo-backend\i(1).m4a"
    
    # 执行转换
    output_file = convert_m4a_to_wav(input_file)
    
    if output_file:
        print(f"转换成功！输出文件: {output_file}")
    else:
        print("转换失败！") 