import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoTokenizer # BitsAndBytesConfig 추가
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel ,TaskType # peft 관련 모듈 추가
import pandas as pd
import shutil
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
import random
import itertools
import gc
import nltk
from rouge_score import rouge_scorer
import jiwer
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import math
from pathlib import Path
import csv
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("Available devices:", torch.cuda.device_count())  # 1개만 보여야 정상
print("Device name:", torch.cuda.get_device_name(0))    # 실제로 GPU 1번 이름이 나옴

SEED = 42

def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # 여러 GPU 사용 시
        # CUDA 연산의 결정론적 실행 설정 (성능에 약간 영향 줄 수 있음)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

set_seeds(SEED)
print(f"모든 라이브러리의 시드가 {SEED}로 고정되었습니다.")

class EEGDataset(Dataset):
    def __init__(self, data_dir="/home/work/skku/hyo/hyo/dataset/word-sentence.parquet"):
        df = pd.read_parquet(data_dir)
        raw = df["eeg"].tolist()
        print(f"[1] 로드된 샘플 개수: {len(raw)}")

        word_arrays = []
        for idx, x in enumerate(raw):
            # 문장 EEG만 있는 경우에도 마지막 요소 포함
            vecs = x[:-1] if len(x) > 1 else x
            print(f"  샘플 {idx} - 원본 길이: {len(x)}, 사용할 word-level EEG 개수: {len(vecs)}")

            words = []
            for v in vecs:
                arr = np.asarray(v, dtype=np.float32).ravel()
                assert arr.ndim == 1, f"벡터 차원 오류: {arr.shape}"
                words.append(arr)
            stacked = np.stack(words, axis=0)  # (Ki, 840)
            print(f"    → 스택 후 shape: {stacked.shape}")
            word_arrays.append(stacked)

        # 전체 단어 수 집계
        all_words = np.vstack(word_arrays)   # (총_단어_수, 840)
        print(f"[2] all_words 스택 shape: {all_words.shape}")

        all_words = np.nan_to_num(
            all_words, nan=0.0, posinf=0.0, neginf=0.0
        )
        mu  = all_words.mean(axis=0, keepdims=True)    # (1, 840)
        std = all_words.std(axis=0,  keepdims=True) + 1e-8
        print(f"[3] mu shape: {mu.shape}, std shape: {std.shape}")

        # 정규화 및 Torch 변환
        self.eeg_seqs = []
        for idx, arr in enumerate(word_arrays):
            normed = (arr - mu) / std                  # (Ki, 840)
            print(f"  샘플 {idx} 정규화 후 shape: {normed.shape}")
            self.eeg_seqs.append(torch.from_numpy(normed))

        # 마지막으로 텍스트
        self.texts = [
            toks[-1] if isinstance(toks, (list, np.ndarray)) else toks
            for toks in df["text"].tolist()
        ]
        print(f"[4] text 샘플 개수: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.eeg_seqs[idx], self.texts[idx]

class ConvEEGEncoder(nn.Module):
    """
    840-dim 벡터를 1×840 시퀀스로 보고 Conv1D 두 층으로 잠재표현 생성
    출력은 (B, latent_dim)
    """
    def __init__(self, input_dim=840, latent_dim=128, hidden=256):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, latent_dim, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)   # 길이 840 → 1 로 압축

    def forward(self, x):           # x: (B, feat)
        x = x.unsqueeze(1)          # (B, 1, 840)
        z = self.conv_stack(x)      # (B, latent_dim, 840)
        z = self.pool(z).squeeze(-1)  # (B, latent_dim)
        return z

class RVQ(nn.Module):
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_quantizers = num_quantizers # 코드북의 개수 (n_q)
        self.num_embeddings = num_embeddings # 각 코드북 내 임베딩(코드워드) 개수 (n_emb, 어휘 크기)
        self.embedding_dim = embedding_dim   # 각 임베딩의 차원 (D, latent_dim과 동일)
        self.commitment_cost = commitment_cost # VQ 손실 계산 시 사용되는 하이퍼파라미터

        # num_quantizers 개수만큼의 코드북(nn.Embedding 레이어)을 리스트로 가짐
        self.codebooks = nn.ModuleList([
            nn.Embedding(self.num_embeddings, self.embedding_dim) for _ in range(self.num_quantizers)
        ])
        # 코드북 가중치 초기화 (선택 사항이지만 일반적으로 수행)
        for i, codebook in enumerate(self.codebooks):
            nn.init.uniform_(codebook.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e): # 입력 z_e의 모양: (B, L, D), 여기서 L=1, D=embedding_dim
        B, L, D = z_e.shape
        z_e_flat = z_e.reshape(-1, D) # (B*L, D) 모양으로 펼침 (여기서는 (B, D)와 동일)

        all_quantized_stages = [] # 각 코드북에서 양자화된 벡터들을 저장할 리스트
        all_indices = []          # 각 코드북에서 선택된 인덱스들을 저장할 리스트
        residual = z_e_flat       # 첫 번째 코드북에 입력될 잔차 (초기에는 z_e_flat 전체)

        # num_quantizers 만큼 반복 (각 코드북에 대해 순차적으로 처리)
        for i in range(self.num_quantizers):
            codebook = self.codebooks[i] # 현재 사용할 코드북

            # 현재 잔차(residual)와 현재 코드북의 모든 임베딩 간의 유클리드 거리 제곱 계산
            # distances 모양: (B*L, num_embeddings)
            distances = torch.sum(residual**2, dim=1, keepdim=True) \
                        - 2 * torch.matmul(residual, codebook.weight.t()) \
                        + torch.sum(codebook.weight**2, dim=1, keepdim=True).t()

            # 가장 가까운 임베딩의 인덱스 찾기
            # current_indices 모양: (B*L)
            current_indices = torch.argmin(distances, dim=1)
            all_indices.append(current_indices) # 현재 코드북의 인덱스 저장

            # 선택된 인덱스를 사용하여 양자화된 벡터(코드워드) 가져오기
            # quantized_vector 모양: (B*L, D)
            quantized_vector = codebook(current_indices)
            # 원래 모양 (B, L, D)로 복원하여 저장 (여기서는 (B, 1, D))
            all_quantized_stages.append(quantized_vector.reshape(B, L, D))

            # 다음 코드북으로 넘길 잔차 계산
            # 중요: quantized_vector에서 그래디언트 흐름을 끊기 위해 .detach() 사용
            residual = residual - quantized_vector.detach()

        # 모든 코드북에서 나온 양자화된 벡터들을 합산 (EEGTran 논문 Figure 2 참조)
        # final_quantized_output 모양: (B, L, D)
        final_quantized_output = torch.stack(all_quantized_stages, dim=0).sum(dim=0)

        # 수집된 인덱스들을 (B, L, num_quantizers) 형태로 쌓음
        # stacked_indices 모양: (B, L, n_q) (여기서는 (B, 1, n_q))
        stacked_indices = torch.stack(all_indices, dim=1).reshape(B, L, self.num_quantizers)

        # 최종 반환값: 합산된 양자화 벡터, 쌓인 인덱스 시퀀스, VQ 손실
        # RVQTokenizer의 forward에서는 이 중 첫 두 개를 zq, indices로 받게 됩니다.
        return final_quantized_output, stacked_indices

class RVQTokenizer(nn.Module):
    def __init__(self,
                 feat=840,
                 latent=128,  # 1024->2048
                 n_q=12,
                 n_emb=512,
                 hidden=256,
                 TOKENIZER_CHECKPOINT_PATH = "/home/work/skku/hyo/hyo/model/rvq_best_model_word_12_512.pt"
                 ):
        super().__init__()
        self.n_q = n_q
        self.n_emb = n_emb
        # 실제 ConvEEGEncoder와 RVQ 모듈이 여기에 와야 함
        self.enc = ConvEEGEncoder(feat, latent, hidden)
        self.rvq = RVQ(num_quantizers=n_q, num_embeddings=n_emb, embedding_dim=latent)

        checkpoint = torch.load(TOKENIZER_CHECKPOINT_PATH, map_location="cpu")
        self.enc.load_state_dict(checkpoint["encoder"])
        for i, cb_weight_tensor in enumerate(checkpoint["codebooks"]):
            self.rvq.codebooks[i].weight.data = cb_weight_tensor

    @torch.no_grad()
    def forward(self, x): # x: (B, 840)
        z = self.enc(x)
        quantized_vector, token_indices = self.rvq(z.unsqueeze(1)) # vq_loss는 무시
        zq = quantized_vector
        indices = token_indices # 모양 (B, 1, n_q)
        # 만약 LLaDA 입력용으로 (B, n_q) 모양의 인덱스를 원한다면 squeeze(1) 필요
        # return zq, indices.squeeze(1)
        return zq, indices # 현재 pasted_content.txt의 주석과 맞추려면 이대로

class UnifiedEEGTextTokenizer:
    def __init__(self,
                rvq_tokenizer_instance,
                llada_text_tokenizer_instance,
                max_seq_length,
                v_text_original,
                eeg_token_length,
                ):

        self.rvq_tokenizer = rvq_tokenizer_instance
        self.llada_text_tokenizer = llada_text_tokenizer_instance
        self.max_seq_length = max_seq_length
        self.v_text_original = v_text_original
        self.eeg_token_length = eeg_token_length


        self.bos_token_id = torch.tensor([self.llada_text_tokenizer.bos_token_id], dtype=torch.long, device=config.system.DEVICE)
        self.eos_token_id = torch.tensor([self.llada_text_tokenizer.eos_token_id], dtype=torch.long, device=config.system.DEVICE)
        self.pad_token_id = self.llada_text_tokenizer.pad_token_id if self.llada_text_tokenizer.pad_token_id is not None else self.llada_text_tokenizer.eos_token_id

        self.user_prompt_intro_ids = self.llada_text_tokenizer.encode(
                "<start_id>user<end_id>\n",
                add_special_tokens=False,
                return_tensors="pt"
            ).squeeze(0).to(config.system.DEVICE)

        self.assistant_prompt_intro_ids = self.llada_text_tokenizer.encode(
                "<eot_id><start_id>assistant<end_id>\n",
                add_special_tokens=False,
                return_tensors="pt"
            ).squeeze(0).to(config.system.DEVICE)

        print(f"Unified Tokenizer Initialized:")
        print(f"  BOS ID: {self.bos_token_id.item()}")
        print(f"  EOS ID: {self.eos_token_id.item()}")
        print(f"  PAD ID: {self.pad_token_id}")
        print(f"  User Prompt Intro IDs ({len(self.user_prompt_intro_ids)} tokens): {self.user_prompt_intro_ids.tolist()}")
        print(f"  Assistant Prompt Intro IDs ({len(self.assistant_prompt_intro_ids)} tokens): {self.assistant_prompt_intro_ids.tolist()}")

    def process_single_sample(self, eeg_seq, assistant_response_text):
        # eeg_seq: (N_words, 840)
        device = config.system.DEVICE

        # 1) 각 word EEG 벡터 → RVQTokenizer → (1,1,n_q) indices → (n_q,) tensor
        local_indices = []
        with torch.no_grad():
            for w in eeg_seq.to(device):                          # w: (840,)
                _, idx = self.rvq_tokenizer(w.unsqueeze(0))       # idx.shape = (1,1,n_q)
                local_indices.append(idx.squeeze(0).squeeze(0))    # (n_q,)
        # 최종 shape: (N_words * n_q,)
        local_indices = torch.cat(local_indices, dim=0)

        # 2) global token IDs: 기존 text vocab_size offset
        global_eeg_ids = (local_indices + self.v_text_original).to(device)  # LongTensor

        # 3) prompt 길이 고정 토큰 수 계산
        fixed_len = (
            len(self.bos_token_id)
          + len(self.user_prompt_intro_ids)
          + len(self.assistant_prompt_intro_ids)
          + len(self.eos_token_id)
        )
        # 남는 공간
        max_resp = self.max_seq_length - fixed_len - global_eeg_ids.numel()
        if max_resp < 1:
            # EEG 토큰이 너무 길면 잘라냅니다
            global_eeg_ids = global_eeg_ids[: self.max_seq_length - fixed_len - 1]
            max_resp = 1

        # 4) 어시스턴트 텍스트 토큰화
        tok = self.llada_text_tokenizer(
            assistant_response_text,
            max_length=max_resp,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        assistant_ids = tok.input_ids.squeeze(0).to(device)

        # 5) input_ids, labels, attention_mask 구성
        input_ids = torch.cat([
            self.bos_token_id,
            self.user_prompt_intro_ids,
            global_eeg_ids,
            self.assistant_prompt_intro_ids,
            assistant_ids,
            self.eos_token_id
        ], dim=0)

        prompt_len = (
            len(self.bos_token_id)
          + len(self.user_prompt_intro_ids)
          + global_eeg_ids.numel()
          + len(self.assistant_prompt_intro_ids)
        )

        # labels: prompt 위치는 -100
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # padding / truncate
        cur_len = input_ids.numel()
        if cur_len < self.max_seq_length:
            pad_len = self.max_seq_length - cur_len
            pad = torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=device)
            input_ids    = torch.cat([input_ids, pad], dim=0)
            labels       = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long, device=device)], dim=0)
            attn_mask    = torch.cat([torch.ones(cur_len, dtype=torch.long, device=device),
                                      torch.zeros(pad_len, dtype=torch.long, device=device)], dim=0)
        else:
            input_ids    = input_ids[:self.max_seq_length]
            labels       = labels[:self.max_seq_length]
            attn_mask    = torch.ones(self.max_seq_length, dtype=torch.long, device=device)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "prompt_lengths": torch.tensor(prompt_len, dtype=torch.long, device=device)
        }

    def build_chat_template_prompt(self, eeg_seq):
        device = config.system.DEVICE
        # ① RVQ → indices
        local_indices = []
        with torch.no_grad():
            for w in eeg_seq.to(device):
                _, idx = self.rvq_tokenizer(w.unsqueeze(0))
                local_indices.append(idx.squeeze(0).squeeze(0))
        global_eeg_ids = (torch.cat(local_indices, dim=0) + self.v_text_original).to(device)

        input_ids = torch.cat([
            self.bos_token_id,
            self.user_prompt_intro_ids,
            global_eeg_ids,
            self.assistant_prompt_intro_ids
        ], dim=0)

        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": torch.ones_like(input_ids).unsqueeze(0),
            "prompt_len": input_ids.numel()
        }

class DataCollatorForEEGTextSFT:
    def __init__(self, unified_tokenizer_instance):
        self.unified_tokenizer = unified_tokenizer_instance

    def __call__(self, batch_of_samples):
        processed_samples = []
        for eeg_tensor, assistant_response_text in batch_of_samples:
            if eeg_tensor is None or assistant_response_text is None:
                continue
            processed_samples.append(self.unified_tokenizer.process_single_sample(eeg_tensor, assistant_response_text))

        if not processed_samples:
            # print("Warning: Collator received no valid samples to batch.")
            # 빈 텐서를 반환하거나 None을 반환하여 학습 루프에서 처리
            return None

        batched_input_ids = torch.stack([s["input_ids"] for s in processed_samples])
        batched_attention_mask = torch.stack([s["attention_mask"] for s in processed_samples])
        batched_labels = torch.stack([s["labels"] for s in processed_samples])
        batched_prompt_lengths = torch.stack([s["prompt_lengths"] for s in processed_samples])

        return {
            "input_ids": batched_input_ids,
            "attention_mask": batched_attention_mask,
            "labels": batched_labels,
            "prompt_lengths": batched_prompt_lengths
        }

def collate_fn_for_evaluation(batch):
    eeg_data_list = [item[0] for item in batch] # 원본 EEG 데이터 리스트
    reference_texts_list = [item[1] for item in batch] # 참조 텍스트 리스트

    # EEG 데이터는 배치 내에서 패딩 없이 리스트 형태로 유지하거나,
    # 만약 모든 EEG 데이터의 길이가 같다면 torch.stack을 사용할 수 있습니다.
    # 여기서는 리스트 형태로 반환하고, 생성 루프에서 개별 처리한다고 가정합니다.
    return {
        "batched_eeg_data": eeg_data_list, 
        "batched_reference_texts": reference_texts_list
    }

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_scores(references, hypotheses):
    """
    references: List[List[List[str]]]  # 각 샘플마다 여러 레퍼런스 문장(토큰화된 리스트)
    hypotheses: List[List[str]]        # 각 샘플마다 생성된 문장(토큰화된 리스트)
    
    반환: {
        "BLEU-1": float,
        "BLEU-2": float,
        "BLEU-3": float,
        "BLEU-4": float
    }
    """
    # 스무딩 함수
    smooth_fn = SmoothingFunction().method1
    
    # 측정할 BLEU weight 설정
    weight_dict = {
        "BLEU-1": (1.0, 0.0, 0.0, 0.0),
        "BLEU-2": (0.5, 0.5, 0.0, 0.0),
        "BLEU-3": (1/3, 1/3, 1/3, 0.0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25),
    }
    
    results = {}
    for name, weights in weight_dict.items():
        scores = []
        for refs, hyp in zip(references, hypotheses):
            # 빈 문자열이나 레퍼런스가 없으면 0점 처리
            if not refs or not refs[0] or not hyp:
                scores.append(0.0)
                continue
            score = sentence_bleu(
                refs,
                hyp,
                weights=weights,
                smoothing_function=smooth_fn
            )
            scores.append(score)
        # 샘플별 점수 평균
        results[name] = float(np.mean(scores)) if scores else 0.0
    
    return results

def calculate_rouge_scores(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            # 빈 문자열 처리 또는 특정 값 할당
            for key in all_scores:
                all_scores[key].append(0.0) # f-measure 기준 0점으로 처리
            continue
        scores = scorer.score(ref, hyp)
        all_scores['rouge1'].append(scores['rouge1'].fmeasure)
        all_scores['rouge2'].append(scores['rouge2'].fmeasure)
        all_scores['rougeL'].append(scores['rougeL'].fmeasure)

    avg_scores = {key: np.mean(values) if values else 0.0 for key, values in all_scores.items()}
    return avg_scores

def calculate_wer(references, hypotheses):
    """
    주어진 정답과 예측에 대해 WER을 계산합니다.
    references: list of strings (각 샘플에 대한 정답 문장 리스트)
    hypotheses: list of strings (각 샘플에 대한 예측 문장 리스트)
    """
    if not references or not hypotheses or len(references) != len(hypotheses):
        return 1.0 # 잘못된 입력 처리, WER은 낮을수록 좋으므로 1.0 (100% 오류) 반환

    # jiwer.compute_measures는 전체 리스트에 대해 집계된 결과를 반환합니다.
    # 개별 문장에 대한 WER을 계산하고 평균내는 것보다, 전체 말뭉치에 대한 WER을 계산하는 것이 일반적입니다.
    measures = jiwer.compute_measures(references, hypotheses)
    return measures['wer']

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

def load_finetuned_eeg_llada(
        base_ckpt: str = "GSAI-ML/LLaDA-8B-Base",
        comp_dir: Path = Path(
            "/home/work/skku/hyo/hyo/checkpoints_v4/best_finetuned_model_components"  # wte.pth, lm_head.pth, adapters/
        ),
        rvq_n_emb: int = 512,
        device: str = "cuda",
        load_in_4bit: bool = True,
):
    # --- ① base LLaDA (4-bit optional) ------------------------------
    bnb_cfg = (BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        if load_in_4bit else None)

    base = AutoModelForCausalLM.from_pretrained(
        base_ckpt, quantization_config=bnb_cfg, torch_dtype="auto",
        trust_remote_code=True).to(device)

    # --- ② vocab resize --------------------------------------------
    v_txt = base.config.vocab_size
    base.resize_token_embeddings(v_txt + rvq_n_emb + 1)   # +1 = EEG-MASK

    # --- ③ embedding / lm_head 복원 -------------------------------
    wte_path = comp_dir / "wte.pth"
    lm_path = comp_dir / "lm_head.pth"

    llada_core = getattr(base, "model", base)          # 1차
    llada_core = getattr(llada_core, "model", llada_core)  # 2차
    #   3-2. transformer 핸들
    transformer = getattr(llada_core, "transformer", llada_core)

    if wte_path.exists():
        base.get_input_embeddings().load_state_dict(torch.load(wte_path, map_location="cpu"))
        print("✔ wte restored")

    if lm_path.exists():
        transformer.ff_out.load_state_dict(torch.load(lm_path, map_location="cpu"))
        print("✔ lm_head (ff_out) restored")

    # --- ④ LoRA attach (trainable=False 로 inference) --------------
    adapter_dir = comp_dir / "adapters"
    if adapter_dir.exists():
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        print("✔ LoRA adapter loaded")
    else:
        model = base
        print("⚠ LoRA adapter not found, base model only")

    model.eval()
    return model

class SystemConfig(BaseModel):
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 0

class PathsConfig(BaseModel):
    DATASET_PATH: str = "/home/work/skku/hyo/hyo/dataset/word-sentence.parquet"
    TOKENIZER_CHECKPOINT_PATH: str = "/home/work/skku/hyo/hyo/model/rvq_best_model_word_12_512.pt"
    MODEL_SAVE_DIR: str = "./saved_models"
    BEST_MODEL_FILENAME: str = "eeg_llada_sft_best_model.pth"
    # LLADA_LOSS_FUNCTION_PATH: str = "/home/ubuntu/llada_loss_function.py" # 필요시 추가
    # MODIFIED_TRAINING_LOOPS_PATH: str = "/home/ubuntu/modified_training_loops.py" # 필요시 추가

class EEGEncoderConfig(BaseModel):
    INPUT_DIM: int = 840
    LATENT_DIM: int = 128 # RVQ의 embedding_dim과 일치해야 함
    HIDDEN_DIM: int = 256

class RVQConfig(BaseModel):
    NUM_QUANTIZERS: int = 12 # RVQTokenizer의 n_q, UnifiedEEGTextTokenizer의 eeg_token_length와 일치
    NUM_EMBEDDINGS: int = 512 # RVQTokenizer의 n_emb
    EMBEDDING_DIM: int = 128 # EEGEncoderConfig의 LATENT_DIM과 일치
    COMMITMENT_COST: float = 0.25

class TokenizerConfig(BaseModel):
    # RVQTokenizer 내부 파라미터 (EEGEncoderConfig, RVQConfig 값으로 대체 가능)
    # UnifiedEEGTextTokenizer 파라미터
    MAX_SEQ_LENGTH: int = 1024 # LLaDA 모델의 최대 컨텍스트 길이 고려
    V_TEXT_ORIGINAL: int = 32000 # LLaMA 텍스트 토크나이저의 어휘 크기 (LLaDA-8B 기준)
    # EEG_TOKEN_LENGTH: int = 12 # RVQConfig.NUM_QUANTIZERS 와 동일
    LLM_MODEL_NAME: str = "GSAI-ML/LLaDA-8B-Base" # LLM 토크나이저 로드용

class ModelConfig(BaseModel):
    LLM_MODEL_NAME: str = "GSAI-ML/LLaDA-8B-Base"
    USE_QLORA: bool = True
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_BIAS: str = "none"
    # LLADA_MASK_TOKEN_ID: Optional[int] = None # 동적으로 설정될 수 있음 (어휘크기 + 1)

class TrainingConfig(BaseModel):
    BATCH_SIZE: int = 4 # GPU 메모리에 따라 조절
    NUM_EPOCHS: int = 10
    START_EPOCH: int = 0
    LEARNING_RATE: float = 1e-4
    GRADIENT_ACCUMULATION_STEPS: int = 4 # BATCH_SIZE * GRAD_ACCUM = 실제 배치 크기
    MAX_GRAD_NORM: float = 1.0
    # SCHEDULER: Optional[str] = None # 예: "StepLR"
    TRAIN_LOG_INTERVAL: int = 1
    PATIENCE_EARLY_STOPPING: int = 3 # 0이면 비활성화

class GenerationConfig(BaseModel):
    RUN_TEST_LOOP_EACH_EPOCH: bool = False
    USE_LLADA_SAMPLING_FOR_GENERATION: bool = True
    MAX_GEN_TOKENS: int = 64
    NUM_SAMPLING_STEPS_GEN: int = 10
    REMASKING_STRATEGY_GEN: str = "low_confidence"
    REMASKING_RATIO_GEN: float = 0.25
    TEMPERATURE_GEN: float = 0.7
    TOP_K_GEN: int = 50
    TOP_P_GEN: float = 0.9
    HF_NUM_BEAMS_GEN: int = 1
    # HF_MAX_LENGTH_GEN: Optional[int] = None # 동적으로 설정 (입력길이 + MAX_GEN_TOKENS)

class ExperimentConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    eeg_encoder: EEGEncoderConfig = Field(default_factory=EEGEncoderConfig)
    rvq: RVQConfig = Field(default_factory=RVQConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    # LLADA_MASK_TOKEN_ID는 동적으로 설정될 수 있으므로, 초기화 후 설정하는 것을 권장
    # 예: config.model.LLADA_MASK_TOKEN_ID = tokenizer.llada_text_tokenizer.vocab_size + 1

config = ExperimentConfig()

comp_dir = Path("/home/work/skku/hyo/hyo/checkpoints_v4/best_finetuned_model_components")
model = load_finetuned_eeg_llada(
            base_ckpt="GSAI-ML/LLaDA-8B-Base",
            comp_dir=comp_dir,
            rvq_n_emb=512,
            device="cuda")

llada_txt_tokenizer = AutoTokenizer.from_pretrained(config.model.LLM_MODEL_NAME)
rvq_eeg_tokenizer = RVQTokenizer()
rvq_eeg_tokenizer = rvq_eeg_tokenizer.to(config.system.DEVICE)
rvq_eeg_tokenizer.eval() # 추론 모드로 설정

v_original = llada_txt_tokenizer.vocab_size
eeg_seq_len = 12 # RVQ_N_Q
model_max_len = 512

unified_eeg_text_tokenizer = UnifiedEEGTextTokenizer(
    rvq_tokenizer_instance=rvq_eeg_tokenizer,
    llada_text_tokenizer_instance=llada_txt_tokenizer,
    max_seq_length=model_max_len,
    v_text_original=v_original,
    eeg_token_length=eeg_seq_len
)

# 1. EEGDataset 인스턴스 생성 (config 사용)
# EEGDataset 클래스 정의는 이미 노트북에 있다고 가정합니다.
eeg_dataset = EEGDataset(data_dir=config.paths.DATASET_PATH)
eeg_dataset = eeg_dataset # 테스트용
num_total_samples = len(eeg_dataset)
indices = list(range(num_total_samples))

# 2. 데이터셋 분할 (config 사용)
# 참고: test_size 값들(현재 0.2 및 0.5)도 config 객체에 추가하여 관리할 수 있습니다.
# 예: config.training.TRAIN_VAL_SPLIT_RATIO, config.training.VAL_TEST_SPLIT_RATIO
train_indices, temp_test_indices = train_test_split(
    indices,
    test_size=0.2, # 전체 데이터 중 20%를 (검증+테스트)용으로 분리
    random_state=config.system.SEED, # config에서 SEED 값 사용
    shuffle=True
)

val_indices, test_indices = train_test_split(
    temp_test_indices,
    test_size=0.5, # (검증+테스트)용 데이터 중 50%를 테스트용으로 분리 (즉, 전체의 10%)
    random_state=config.system.SEED, # config에서 SEED 값 사용
    shuffle=True
)

# 각 Subset 생성
train_dataset = Subset(eeg_dataset, train_indices)
val_dataset = Subset(eeg_dataset, val_indices)
test_dataset = Subset(eeg_dataset, test_indices)

print(f"전체 데이터셋 크기: {num_total_samples}")
print(f"학습 세트 크기: {len(train_dataset)} (전체의 {len(train_dataset)/num_total_samples:.2%})")
print(f"검증 세트 크기: {len(val_dataset)} (전체의 {len(val_dataset)/num_total_samples:.2%})")
print(f"테스트 세트 크기: {len(test_dataset)} (전체의 {len(test_dataset)/num_total_samples:.2%})")

# DataCollatorForEEGTextSFT 인스턴스 생성 (이전에 data_collator 로 정의되었다고 가정)
data_collator = DataCollatorForEEGTextSFT(unified_eeg_text_tokenizer)

# 3. DataLoader 생성 (config 사용)
# 학습 데이터 로더
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.training.BATCH_SIZE, # config에서 BATCH_SIZE 값 사용
    collate_fn=collate_fn_for_evaluation,
    shuffle=True,
    num_workers=config.system.NUM_WORKERS, # config에서 NUM_WORKERS 값 사용
    pin_memory= False
)

# 검증 데이터 로더
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.training.BATCH_SIZE, # config에서 BATCH_SIZE 값 사용
    collate_fn=collate_fn_for_evaluation,
    shuffle=False,
    num_workers=config.system.NUM_WORKERS, # config에서 NUM_WORKERS 값 사용
    pin_memory= False
)

# 테스트 데이터 로더
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.training.BATCH_SIZE, # config에서 BATCH_SIZE 값 사용
    collate_fn=collate_fn_for_evaluation,
    shuffle=False,
    num_workers=config.system.NUM_WORKERS, # config에서 NUM_WORKERS 값 사용
    pin_memory= False
)

print(f"\n학습 데이터로더 배치 수: {len(train_dataloader)}")
print(f"검증 데이터로더 배치 수: {len(val_dataloader)}")
print(f"테스트 데이터로더 배치 수: {len(test_dataloader)}")


print("\n데이터 로딩 및 분할 (config 기반) 완료.")

# ───── 샘플링 하이퍼파라미터 ─────
GEN_STEPS   = 64    # diffusion 스텝
GEN_LEN     = 128        # 생성 토큰 수
BLOCK_LEN   = 64
TEMP        = 0.3
CFG_SCALE   = 0.9
REMASKING   = "low_confidence"

device = next(model.parameters()).device        # 모델이 올라간 GPU/CPU

model.eval()
all_hyp, all_ref = [], []

# ─── 결과 저장용 ───
out_dir      = Path("/home/work/skku/hyo/hyo/generate")
out_dir.mkdir(exist_ok=True)
csv_path     = out_dir / "eeg_gen_results.csv"
metrics_path = out_dir / "final_metrics.json"

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="🔮  Generating from EEG"):
        eeg_list = batch["batched_eeg_data"]            # list[Tensor(840)]
        ref_list = batch["batched_reference_texts"]     # list[str]

        for eeg_tensor, ref_text in zip(eeg_list, ref_list):
            # 1) EEG  ➜  RVQ  ➜  prompt
            prompt_pack = unified_eeg_text_tokenizer.build_chat_template_prompt(eeg_tensor)
            prompt_ids  = prompt_pack["input_ids"].to(device)          # (1, T)
            prompt_len  = prompt_pack["prompt_len"]
            # EEG 토큰 id 가 서로 다른지
            # print(prompt_ids[0, :20])          # 다른 EEG sample 에서 값이 달라야 함

            # # 어댑터 로드됐는지
            # print(model.peft_config.keys())    # LoRA 설정 dict 가 비어 있으면 어댑터 미적용

            # 2) LLaDA diffusion sampling
            out_ids = generate(
                model,                                  # ← 바로 model 전달
                prompt = prompt_ids,         # (T,)
                steps  = GEN_STEPS,
                gen_length = GEN_LEN,
                block_length = BLOCK_LEN,
                temperature = TEMP,
                cfg_scale   = CFG_SCALE,
                remasking   = REMASKING
            )

            # 3) 디코딩 (prompt 이후만)
            gen_txt = unified_eeg_text_tokenizer.llada_text_tokenizer.batch_decode(
                out_ids[:, prompt_len:], skip_special_tokens=True
            )[0].strip()
    
            # 4) 즉시 CSV로 flush (중간에 끊겨도 데이터 보존)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ref_text, gen_txt])

            all_ref.append(ref_text)
            all_hyp.append(gen_txt)

    # ── 최종 지표 계산 & 저장 ──
    bleu  = calculate_bleu_scores(all_ref, all_hyp)
    rouge = calculate_rouge_scores(all_ref, all_hyp)
    wer   = calculate_wer(all_ref, all_hyp)
    metrics = {"BLEU": bleu, "ROUGE": rouge, "WER": wer}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("📝  finished! metrics saved →", metrics_path)