import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def plot_segment(mel_spec:np.ndarray, note_times:list, start_sec:int, end_sec:int, sr:int, HOP_LENGTH:int):
    plt.figure(figsize=(15, 5))
    
    # 将分贝转换回可视化亮度
    db_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 绘制频谱图
    librosa.display.specshow(db_spec, sr=sr, hop_length=HOP_LENGTH, 
                             x_axis='time', y_axis='mel', fmax=8000)
    
    # 绘制你的 note_time_list
    plt.vlines(note_times, 0, 8000, color='g', linestyle='-', linewidth=1.5, label='Actual Notes')
    
    # 关键步骤：限制显示的时间范围
    plt.xlim(start_sec, end_sec) 
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Check Alignment: {start_sec}s to {end_sec}s")
    plt.legend(loc="upper right")
    plt.show()

def click_with_track(note_frames:np.ndarray, sample_rate:int, HOP_LENGTH:int, raw_track:np.ndarray):
    clicks = librosa.clicks(frames=note_frames, sr=sample_rate, hop_length=HOP_LENGTH, length=len(raw_track))
    clicks = clicks * 3
    # 叠加到原曲
    y_with_clicks = raw_track + clicks
    # 保存出来听一下，如果“嘀”声和鼓点重合，说明你的数据稳了
    import soundfile as sf
    sf.write('test_alignment.wav', y_with_clicks, sample_rate)