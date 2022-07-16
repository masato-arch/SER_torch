'''
このファイルの使い方
import Load_IEMOCAP
dataset = Load_IEMOCAP.create_dataset()

これでデータセットが返される
'''

import os
import re
from statistics import mode
import collections
import librosa
import math
import audeer
import pandas as pd


'''前準備: グローバルパスの定義'''

# wavファイルがセッションごとに整理されてるディレクトリへのパス
session_dirs = ['../../IEMOCAP_full_release/' + 'Session' + str(i) + '/' + 'sentences/wav/' for i in range(1, 6)]
# ラベルがセッションごとに整理されてるディレクトリへのパス
label_dirs = ['../../IEMOCAP_full_release/' + 'Session' + str(i) + '/dialog/EmoEvaluation/Categorical/' for i in range(1, 6)]

''' 1つ1つのwavファイルへのパスを取る '''

# wavファイルのパスのペアレントディレクトリを取る
wav_file_paths_p = []
for ses_d in session_dirs:
    sentence_dirs = sorted(os.listdir(ses_d))
    for sent_d in sentence_dirs:
        wav_file_paths_p.append(os.path.join(ses_d, sent_d))
       
#不要な.DS_Storeとかいうのが出てきたので削除
for path in wav_file_paths_p:
    if 'DS_Store' in path:
        wav_file_paths_p.remove(path)
        
# 1つ1つのwavファイルへのパスを取る
wav_file_paths = []
for path in wav_file_paths_p:
    filenames = sorted(os.listdir(path))
    for f in filenames:
        wav_file_paths.append(os.path.join(path, f))
        
# wav以外の拡張子のファイルが含まれてたら削除
for path in wav_file_paths:
    base, ext = os.path.splitext(path)
    if not ext == '.wav':
        wav_file_paths.remove(path)
        
# 拡張子を除いたverのファイル名も取っておく
filenames = [audeer.basename_wo_ext(p) for p in wav_file_paths]

# ラベル付けファイル1つ1つへのpathをとる
label_file_paths = []
for label_dir in label_dirs:
    label_files_ = sorted(os.listdir(label_dir))
    for f in label_files_:
        base, ext = os.path.splitext(f)
        if ext == '.txt':
            label_file_paths.append(os.path.join(label_dir, f))
            
'''wavファイルのラベルを取得する'''

# 1つ1つの発話ののラベルはスピーチダイアログごとにtxtファイルを作って管理されている
# スピーチダイアログの名前とwavのペアレントディレクトリの名前は一致する

# 上位パスを排除して名前だけのwav_file_path_pを作る
wav_file_paths_pp = []

for path in wav_file_paths_p:
    wav_file_paths_pp.append(path[path.rfind('/') + 1: ])

# list内のkeyを含む要素を全て返すメソッド
def index_multi(lst, key):
    idxes = []
    for i in range(len(lst)):
        # print(f'examining {lst[i]} for key:{key}')
        if key in lst[i]:
            idxes.append(i)
    return idxes

# スピーチダイアログごとに評価ファイルを分けていく
# 1つのダイアログに複数の評価者がいる
label_file_indexes = []
for dirname in wav_file_paths_pp:
    idxes = index_multi(label_file_paths, dirname)
    label_file_indexes.append(idxes)

# ファイル名から話者idを抽出するディクショナリ
speaker_mapping = {
    'Ses01M' : 1,
    'Ses01F' : 2, 
    'Ses02M': 3,
    'Ses02F' :4,
    'Ses03M': 5,
    'Ses03F': 6,
    'Ses04M': 7,
    'Ses04F': 8,
    'Ses05M': 9,
    'Ses05F': 10
}

def identify_speaker(filename):
    session = filename[:5]
    sex = filename[-4]
    return session + sex

# 各音声ファイルごとにラベルを作っていく
# 同じダイアログに対する複数の評価者のラベル付を同時に読み込んで多数決を行う
raw_labels = []
speakers = []
filenames = []
for i, indexes in enumerate(label_file_indexes):
    # ここで1ダイアログの処理
    # それぞれの評価ファイルのラベルを保存しておくグローバルリスト
    global_labels = []
    
    # それぞれの評価ファイルのラベルを読み込んでglobal_labelsに保存する
    for index in indexes: 
        # ここで1ファイルの処理
        file = open(label_file_paths[index], 'r')
        local_labels = []
        # print(f'#{display_filename(wav_file_paths_pp[i])}: opening {display_filename(label_file_paths[index])}')
        for line in file:
            label = re.split('[:;()]', line)
            label = label[:-1]
            local_labels.append(label)
        global_labels.append(local_labels)
        
    for file_id in range(len(global_labels[0])):
        raw_label = [global_labels[i][file_id][1] for i in range(len(global_labels))]
        filename = global_labels[0][file_id][0][:-1]
        filenames.append(filename)
        speakers.append(speaker_mapping[identify_speaker(filename)])
        raw_labels.append(raw_label)
        
# Neutral state をNeutralに表記変え
for label in raw_labels:
    for i in range(len(label)):
        if label[i] == 'Neutral state':
            label[i] = 'Neutral'
        
'''ExcitedをHappinessにマージする'''

def exc_hap_marge(label):
    label_ = label.copy()
    for i in range(len(label_)):
        if label_[i] == 'Excited':
            label_[i] = 'Happiness'
    return label_

ex_hp_marged = [exc_hap_marge(l) for l in raw_labels]

'''3評価者間でラベル付が割れている音声を除く'''

def unagreed(label):
    return len(label) == len(collections.Counter(label))

agreed = [label for label in ex_hp_marged if not unagreed(label)]
agreed_speakers = [speakers[i] for i in range(len(speakers)) if not unagreed(ex_hp_marged[i])]
file_paths_agreed = [wav_file_paths[i] for i in range(len(wav_file_paths)) if not unagreed(ex_hp_marged[i])]
filenames_agreed = [filenames[i] for i in range(len(filenames)) if not unagreed(ex_hp_marged[i])]


'''多数決を取ってラベルを1つに決定する'''

maj_labels = [mode(label) for label in agreed]

'''必要な感情ラベルだけ取ってくる'''

files_selected = []
filenames_selected = []
speakers_selected = []
labels_selected = []

necessary_labels = ['Anger', 'Neutral', 'Sadness', 'Happiness']
for i in range(len(maj_labels)):
    if maj_labels[i] in necessary_labels:
        files_selected.append(file_paths_agreed[i])
        filenames_selected.append(filenames_agreed[i])
        speakers_selected.append(agreed_speakers[i])
        labels_selected.append(maj_labels[i])

datas = []
for file in files_selected:
    x, fs = librosa.load(file, sr=16000)
    datas.append([x, fs])

class dataset:
    def __init__(self, datas=None, speakers=None, labels=None, speaker_mapping=None, filenames=None):
        self.datas = datas
        self.speakers = speakers
        self.speaker_mapping = speaker_mapping
        self.labels = labels
        self.filenames = filenames

def create_dataset():
    return dataset(datas, speakers_selected, labels_selected, speaker_mapping, filenames_selected)