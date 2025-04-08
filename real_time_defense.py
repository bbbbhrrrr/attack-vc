import os
import argparse
import time
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
from threading import Thread
from queue import Queue
import matplotlib.pyplot as plt

from vsmask import VSMask

def list_audio_devices():
    """列出所有可用的音频设备"""
    print("\n可用的音频设备:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (输入通道: {device['max_input_channels']}, 输出通道: {device['max_output_channels']})")
    print()

def record_audio(input_device, duration, sample_rate, output_file):
    """录制音频并保存到文件"""
    print(f"正在录制 {duration} 秒的音频...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=input_device)
    sd.wait()
    sf.write(output_file, audio, sample_rate)
    print(f"音频已保存到 {output_file}")
    return audio.flatten()

def play_audio(audio, sample_rate, output_device):
    """播放音频"""
    print("正在播放音频...")
    sd.play(audio, sample_rate, device=output_device)
    sd.wait()
    print("播放完成")

def real_time_protection(
    model_dir: str,
    vsmask_model_path: str,
    input_device: int,
    output_device: int,
    duration: float = 30.0,
    epsilon: float = 0.1
):
    """
    实时语音保护
    
    Args:
        model_dir: 语音合成模型目录
        vsmask_model_path: VSMask模型路径
        input_device: 输入设备ID
        output_device: 输出设备ID
        duration: 持续时间(秒)
        epsilon: 扰动最大幅度
    """
    # 初始化VSMask
    vsmask = VSMask(
        model_dir=model_dir,
        epsilon=epsilon
    )
    
    # 加载VSMask模型
    vsmask.load_vsmask_model(vsmask_model_path)
    
    # 获取设备采样率
    device_info = sd.query_devices(input_device, 'input')
    sample_rate = int(device_info['default_samplerate'])
    
    # 创建音频处理队列
    q_in = Queue()
    q_out = Queue()
    
    # 设置运行标志
    running = True
    
    # 计算块大小
    hop_length = vsmask.config["preprocess"]["hop_length"]
    block_size = vsmask.input_frames * hop_length // 4  # 减小块大小以降低延迟
    
    # 音频输入回调函数
    def input_callback(indata, frames, time, status):
        if status:
            print(f"输入状态: {status}")
        q_in.put(indata.copy())
    
    # 音频输出回调函数
    def output_callback(outdata, frames, time, status):
        if status:
            print(f"输出状态: {status}")
        
        if not q_out.empty():
            outdata[:] = q_out.get()
        else:
            outdata[:] = np.zeros_like(outdata)
    
    # 处理线程
    def processing_thread():
        # 创建音频缓冲区
        buffer_duration = 1.0  # 缓冲区长度(秒)
        buffer_size = int(buffer_duration * sample_rate)
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_pos = 0
        
        # 预处理帧大小
        frame_size = vsmask.input_frames * hop_length
        output_size = vsmask.output_frames * hop_length
        
        # 处理循环
        while running:
            # 从输入队列获取音频数据
            if q_in.empty():
                time.sleep(0.001)
                continue
            
            indata = q_in.get().flatten()
            
            # 将新数据添加到缓冲区
            if buffer_pos + len(indata) <= buffer_size:
                audio_buffer[buffer_pos:buffer_pos+len(indata)] = indata
                buffer_pos += len(indata)
            else:
                # 缓冲区已满，移动数据并添加新数据
                shift = len(indata)
                audio_buffer[:-shift] = audio_buffer[shift:]
                audio_buffer[-shift:] = indata
                buffer_pos = buffer_size
            
            # 如果缓冲区中有足够的数据进行处理
            if buffer_pos >= block_size:
                # 从缓冲区提取音频块并应用保护
                audio_block = audio_buffer[:block_size].copy()
                
                # 简化的保护处理
                # 在实际应用中，这里应该使用VSMask模型为该块生成扰动
                # 但为了简化，我们只添加一个预设的扰动
                
                # 生成噪声扰动
                noise = np.random.randn(block_size) * epsilon * 0.3
                
                # 应用频率权重（简化处理）
                freq_weights = np.ones(block_size)
                
                # 添加频率加权噪声
                protected_block = audio_block + noise * freq_weights
                
                # 将保护后的块放入输出队列
                q_out.put(protected_block.astype(np.float32))
                
                # 更新缓冲区
                audio_buffer[:-block_size] = audio_buffer[block_size:]
                buffer_pos -= block_size
    
    # 创建并启动处理线程
    thread = Thread(target=processing_thread)
    thread.daemon = True
    thread.start()
    
    # 打开输入输出流
    with sd.InputStream(device=input_device, channels=1, callback=input_callback, 
                        blocksize=block_size, samplerate=sample_rate):
        with sd.OutputStream(device=output_device, channels=1, callback=output_callback,
                            blocksize=block_size, samplerate=sample_rate):
            
            print(f"实时保护已启动，持续 {duration} 秒...")
            print("请开始说话...")
            
            # 等待指定时间
            sd.sleep(int(duration * 1000))
    
    # 停止处理
    running = False
    thread.join(timeout=1.0)
    
    print("实时保护已停止")

def main():
    parser = argparse.ArgumentParser(description="VSMask实时语音保护")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 列出设备命令
    list_parser = subparsers.add_parser("list", help="列出可用音频设备")
    
    # 实时保护命令
    realtime_parser = subparsers.add_parser("realtime", help="实时语音保护")
    realtime_parser.add_argument("--model_dir", type=str, required=True, help="语音合成模型目录")
    realtime_parser.add_argument("--vsmask_model", type=str, required=True, help="VSMask模型路径")
    realtime_parser.add_argument("--input_device", type=int, required=True, help="输入设备ID")
    realtime_parser.add_argument("--output_device", type=int, required=True, help="输出设备ID")
    realtime_parser.add_argument("--duration", type=float, default=30.0, help="持续时间(秒)")
    realtime_parser.add_argument("--epsilon", type=float, default=0.1, help="扰动最大幅度")
    
    # 录制测试命令
    record_parser = subparsers.add_parser("record", help="录制音频测试")
    record_parser.add_argument("--input_device", type=int, required=True, help="输入设备ID")
    record_parser.add_argument("--output_file", type=str, default="recorded.wav", help="输出文件路径")
    record_parser.add_argument("--duration", type=float, default=5.0, help="录制时间(秒)")
    record_parser.add_argument("--sample_rate", type=int, default=16000, help="采样率")
    
    # 播放测试命令
    play_parser = subparsers.add_parser("play", help="播放音频测试")
    play_parser.add_argument("--input_file", type=str, required=True, help="输入文件路径")
    play_parser.add_argument("--output_device", type=int, required=True, help="输出设备ID")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_audio_devices()
    
    elif args.command == "realtime":
        real_time_protection(
            model_dir=args.model_dir,
            vsmask_model_path=args.vsmask_model,
            input_device=args.input_device,
            output_device=args.output_device,
            duration=args.duration,
            epsilon=args.epsilon
        )
    
    elif args.command == "record":
        record_audio(
            input_device=args.input_device,
            duration=args.duration,
            sample_rate=args.sample_rate,
            output_file=args.output_file
        )
    
    elif args.command == "play":
        audio, sample_rate = sf.read(args.input_file)
        play_audio(
            audio=audio,
            sample_rate=sample_rate,
            output_device=args.output_device
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
