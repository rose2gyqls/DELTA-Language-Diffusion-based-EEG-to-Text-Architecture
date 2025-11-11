import pandas as pd
import os
import numpy as np

# 경로 설정
word_dir = '/workspace/dataset/word'
sent_dir = '/workspace/dataset'
save_dir = '/workspace/hyo/dataset'
os.makedirs(save_dir, exist_ok=True)

# 태스크 리스트 (실제 파일명에 맞게 수정)
tasks = ['1-task1-SR-dataset', '1-task2-NR-dataset', '1-task3-TSR-dataset', '2-task1-NR-dataset', '2-task2-TSR-dataset']

for task in tasks:
    word_path = os.path.join(word_dir, f'{task}.parquet')
    sent_path = os.path.join(sent_dir, f'{task}.parquet')
    
    df_word = pd.read_parquet(word_path)
    df_sent = pd.read_parquet(sent_path)
    
    sent2eeg = dict(zip(df_sent['text'], df_sent['eeg']))
    
    rows = []
    for i, row in df_word.iterrows():
        sentence = row['Sentence']
        word_eeg_list = row['EEG']
        sentence_eeg = sent2eeg.get(sentence)
        
        # 단어 리스트 및 word-level eeg 리스트 추출
        words = [w['content'] for w in word_eeg_list]
        word_eegs = [list(np.array(w['vector'], dtype=np.float32)) for w in word_eeg_list]
        sentence_eeg = list(np.array(sentence_eeg, dtype=np.float32))
        
        # 마지막에 sentence-level 정보 추가
        words_combined = words + [sentence]
        eegs_combined = word_eegs + [sentence_eeg]
        
        rows.append({
            'eeg': eegs_combined,
            'text': words_combined
        })
    df_combined = pd.DataFrame(rows)
    
    # 저장
    save_path = os.path.join(save_dir, f'{task}.parquet')
    df_combined.to_parquet(save_path, index=False)
    print(f'{task} 저장 완료: {save_path}')