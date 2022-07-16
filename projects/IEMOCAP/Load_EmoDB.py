'''
このファイルの使い方
import Load_EmoDB
dataset = Load_EmoDB.create_dataset()
'''

import numpy as np
import librosa
import librosa.display
import audeer
import wave
import os
import audformat
import soundfile as sf
import audiofile as af

#ファイル名から特定の情報を返すメソッド
def parse_names(names, from_i, to_i, is_number=False, mapping=None):
    for name in names:
        key = name[from_i:to_i]
        if is_number:
            key = int(key)
        yield mapping[key] if mapping else key

# wavがあるディレクトリへのpath
Emo_DB_dir = '../../download/wav/'

# wavファイル1つ1つへのpath
files = sorted([os.path.join(Emo_DB_dir, f) for f in os.listdir(Emo_DB_dir)])

# 拡張子と絶対パスを除いたファイル名
names = [audeer.basename_wo_ext(f) for f in files]

male = audformat.define.Gender.MALE
female = audformat.define.Gender.FEMALE
language = audformat.utils.map_language('de')

# ファイル名と話者の間のディクショナリ
speaker_mapping = {
    3: {'gender': male, 'age': 31, 'language': language},
    8: {'gender': female, 'age': 34, 'language': language},
    9: {'gender': female, 'age': 21, 'language': language},
    10: {'gender': male, 'age': 32, 'language': language},
    11: {'gender': male, 'age': 26, 'language': language},
    12: {'gender': male, 'age': 30, 'language': language},
    13: {'gender': female, 'age': 32, 'language': language},
    14: {'gender': female, 'age': 35, 'language': language},
    15: {'gender': male, 'age': 25, 'language': language},
    16: {'gender': female, 'age': 31, 'language': language},
}

# ファイル名と感情ラベルの間のディクショナリ
emotion_mapping = {
    'W': 'Anger',
    'L': 'Boredom',
    'E': 'Disgust',
    'A': 'Fear',
    'F': 'Happiness',
    'T': 'Sadness',
    'N': 'Neutral',
}

# ファイルごとの話者
speakers = list(parse_names(names, from_i=0, to_i=2, is_number=True))

# ファイルごとの感情ラベル
emotions = list(parse_names(names, from_i=5, to_i=6, mapping=emotion_mapping))

# wavファイルをロード
datas = []
for f in files:
    x, fs = librosa.load(f, sr=16000)
    datas.append([x, fs])

necessary_labels = ['Neutral', 'Happiness', 'Sadness', 'Anger']

class dataset:
    def __init__(self, datas, speakers, emotions, speaker_mapping, emotion_mapping):
        self.datas = datas
        self.speakers = speakers
        self.emotions = emotions
        self.speaker_mapping = speaker_mapping
        self.emotion_mapping = emotion_mapping
        
    def selected_dataset(self, necessary_labels):
        selected_datas = []
        selected_labels = []
        selected_speakers = []
        for i in range(len(self.emotions)):
            if self.emotions[i] in necessary_labels:
                selected_datas.append(self.datas[i])
                selected_labels.append(self.emotions[i])
                selected_speakers.append(self.speakers[i])
                
        return selected_datas, selected_labels, selected_speakers
    
def create_dataset():
    ds = dataset(datas, speakers, emotions, speaker_mapping, emotion_mapping)
    selected_datas, selected_labels, selected_speakers = ds.selected_dataset(necessary_labels)
    return dataset(selected_datas, selected_speakers, selected_labels, speaker_mapping, emotion_mapping)