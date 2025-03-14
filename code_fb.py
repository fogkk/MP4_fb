import numpy as np
from moviepy.editor import VideoFileClip
import pandas as pd
from tqdm import tqdm

def calculate_db(signal):
    """计算音频信号的分贝值"""
    rms = np.sqrt(np.mean(signal**2))
    if rms > 0:
        return 20 * np.log10(rms) 
    else:
        return -np.inf 

def extract_audio_db_from_mp4_try(video_path):
    """从MP4文件中提取音频并计算每帧的分贝值"""
    video = VideoFileClip(video_path)
    audio = video.audio
    fps = audio.fps 
    audio_array = audio.to_soundarray(fps = fps, nbytes=2) 
    # num_frames = audio_array.shape 
    a_num = 44100 * 2
    b_num = 44100 
    # breakpoint()
    db_values = [] 
    for i in tqdm(range(b_num, a_num)): 
        frame = audio_array[i]
        db = calculate_db(frame)
        # db取绝对值
        db = abs(db)
        db_values.append(db)
        
    df = pd.DataFrame(db_values, columns=['decibel'])
    # df.index = range(b_num, b_num + len(df))
    df.index = range(b_num, b_num + len(df)) 
    # df.index = range(50000000,50000000 + len(df)) 
    df.to_csv('db_values_with_index.csv', index=True)
    return df

# 给定 fps = audio.fps，这表示 audio 对象的采样率被赋值给了变量 fps。如果 fps = 44000，这意味着音频的采样率是 44000 赫兹（Hz），即每秒有 44000 个样本。

def extract_audio_db_from_mp4(video_path):
    """从MP4文件中提取音频并计算每帧的分贝值"""
    video = VideoFileClip(video_path)
    audio = video.audio
    fps = audio.fps 
    audio_array = audio.to_soundarray(fps = fps, nbytes=2) 
    num_frames_f = audio_array.shape

    print(f"the mp4 which change space is {num_frames_f}")
    a_num, b_num = num_frames_f 
    print(f"the mp4 range is a_num:{a_num}, b_num:{b_num}")
    db_values = []

    df = pd.DataFrame(audio_array) 
    num_groups = len(df) // 44100
    index_array = np.arange(len(df))
    len_array = len(index_array)
    
    mean_df = [df.iloc[index_array[i:i + fps]].mean() for i in range(0,len_array , fps)]
    breakpoint()
    # reset_index
    mean_df = pd.DataFrame(mean_df).reset_index(drop=True)
    long = len(mean_df)
    for i in tqdm(range(long)):
        frame = mean_df.iloc[i]
        db = calculate_db(frame)
        # db取绝对值
        db_values.append(db)

    df_c = pd.DataFrame(db_values, columns=['decibel'])
    df_c.to_csv('db_values.csv', index=True)


# 使用示例
video_path = r'E:\MP4_fb\mp4\try.MP4'
db_values = extract_audio_db_from_mp4(video_path)
# db_values = extract_audio_db_from_mp4_try(video_path)
