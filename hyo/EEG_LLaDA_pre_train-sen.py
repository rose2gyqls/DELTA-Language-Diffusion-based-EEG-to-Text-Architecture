import torch
import torch.nn as nn
import torch.nn.functional as F # F.cross_entropy를 위해 추가
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig # BitsAndBytesConfig 추가
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType # peft 관련 모듈 추가
import pandas as pd
import shutil
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
import random
import itertools
import gc
from tqdm import tqdm
import time

# 시드 값 설정 (원하는 정수 값으로 설정)
SEED = 42

def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # 여러 GPU 사용 시

set_seeds(SEED)

class EEGDataset(Dataset):
    def __init__(self,
                 data_dir = "/home/work/skku/hyo/hyo/dataset/sentence.parquet"):
        df = pd.read_parquet(data_dir)
        eeg_vecs = df["eeg"].to_numpy()

        arr = np.stack(eeg_vecs).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mu, std = arr.mean(0, keepdims=True), arr.std(0, keepdims=True)+1e-8
        self.eeg_arr = (arr - mu) / std      # 정규화
        self.text_arr = df["text"].to_numpy() # 텍스트 데이터
        self.data = list(zip(torch.tensor(self.eeg_arr), self.text_arr))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
                 TOKENIZER_CHECKPOINT_PATH = "/home/work/skku/hyo/hyo/model/rvq_best_model_sen_512.pt"
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

def forward_process_eeg(original_eeg_ids, mask_eeg_token_id = 126336, eps=1e-3):
    # original_eeg_ids: (B, n_q) 모양의 EEG 토큰 ID
    # mask_eeg_token_id: 우리가 정의한 EEG 마스크 토큰 ID (예: RVQ_N_EMB)
    b, l = original_eeg_ids.shape

    # 각 배치 샘플별로 랜덤한 t 값을 생성 (0~1)
    # LLaDA 코드는 t를 (b)로 만들지만, 논문 Figure 2a는 t ~ U(0,1)로 단일 값을 의미하기도 합니다.
    # 여기서는 LLaDA 코드 스타일을 따라 배치별 t를 사용합니다.
    t_per_sample = torch.rand(b, device=original_eeg_ids.device)

    # p_mask 계산: 각 샘플의 t 값에 따라 해당 샘플 내 모든 토큰에 적용될 마스킹 확률
    # p_mask_per_sample의 모양: (b, 1)
    p_mask_per_sample = (1 - eps) * t_per_sample + eps
    # p_mask_for_tokens의 모양: (b, l)
    p_mask_for_tokens = p_mask_per_sample.unsqueeze(-1).repeat(1, l)

    # 각 토큰 위치별로 마스킹 여부 결정
    # noise_for_masking의 모양: (b, l)
    noise_for_masking = torch.rand((b, l), device=original_eeg_ids.device)
    masked_indices = noise_for_masking < p_mask_for_tokens # True면 마스크

    # 마스크된 입력 생성 (noisy_batch 역할)
    masked_eeg_ids_for_input = torch.where(masked_indices, mask_eeg_token_id, original_eeg_ids)

    return masked_eeg_ids_for_input, masked_indices # p_mask는 직접 필요 없으므로 반환 안 함 (필요시 추가)

class EEG_LLaDA_MLM(nn.Module):
    def __init__(self, llada_model_name, rvq_n_emb, use_qlora=True, qlora_config_params=None):
        super().__init__()
        self.rvq_n_emb = rvq_n_emb
        self.llada_model_name = llada_model_name

        bnb_config = None
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, # 또는 torch.float16
                bnb_4bit_use_double_quant=True,
            )

        # LLaDA 모델 로드 (양자화 설정 적용)
        self.llada_model = AutoModelForCausalLM.from_pretrained(
            llada_model_name,
            quantization_config=bnb_config if use_qlora else None,
            torch_dtype=torch.bfloat16 if use_qlora and bnb_config else "auto", # 양자화 시 bfloat16 사용 권장
            trust_remote_code=True,
            # device_map="auto" # 여러 GPU 사용 시 또는 메모리 최적화 시 고려
        )
        self.llada_hidden_size = self.llada_model.config.hidden_size
        model_dtype = self.llada_model.dtype

        self.v_text = self.llada_model.config.vocab_size
        num_new_eeg_tokens = self.rvq_n_emb + 1
        new_total_vocab_size = self.v_text + num_new_eeg_tokens
        print(f"Original vocab size: {self.v_text}")
        print(f"Resizing token embeddings to: {new_total_vocab_size}")
        self.llada_model.resize_token_embeddings(new_total_vocab_size)        
        self.global_mask_eeg_token_id = self.v_text + self.rvq_n_emb

# # ----------------------------------------------  FIX  ---------------------------------------------- #
#         #  text 쪽은 0 … self.v_text-1,                  
#         #  EEG 토큰은 self.eeg_token_offset … new_total_vocab_size-2,  
#         #  마스크 토큰은 **맨 마지막 행 번호**.
#         self.eeg_token_offset = self.v_text              # = 원래 vocab_size
#         self.global_mask_eeg_token_id = new_total_vocab_size - 1 # = 마지막 index (= 행 수-1)
# # ---------------------------------------------------------------------------------------------------- #        

        # QLoRA 적용
        if use_qlora:
            # 모델을 k-bit 학습용으로 준비 (양자화된 모델에 필요)
            #self.llada_model = prepare_model_for_kbit_training(self.llada_model)

            # LoRA 설정 정의
            # target_modules는 모델마다 다를 수 있으므로 확인 필요 (아래 설명 참조)
            default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if qlora_config_params and "target_modules" in qlora_config_params:
                target_modules = qlora_config_params["target_modules"]
            else:
                target_modules = default_target_modules

            lora_config = LoraConfig(
                r=qlora_config_params.get("r", 16) if qlora_config_params else 16, # LoRA rank
                lora_alpha=qlora_config_params.get("lora_alpha", 32) if qlora_config_params else 32, # Alpha scaling
                target_modules=target_modules,
                lora_dropout=qlora_config_params.get("lora_dropout", 0.05) if qlora_config_params else 0.05,
                bias="none", # LoRA는 보통 bias를 학습하지 않음
                task_type=TaskType.CAUSAL_LM, # Causal LM 작업용
            )
            self.llada_model = get_peft_model(self.llada_model, lora_config)
            print("QLoRA applied to LLaDA model.")

            print("Making input embeddings trainable for newly added tokens...")
            if hasattr(self.llada_model, 'base_model'): # PeftModel 경우
                embedding_layer = self.llada_model.base_model.get_input_embeddings()
            else: # 일반 모델 경우 (get_peft_model 이전)
                embedding_layer = self.llada_model.get_input_embeddings()

            for param in embedding_layer.parameters():
                param.requires_grad = True
            print("Input embeddings are now trainable.")

            self.llada_model.print_trainable_parameters() # 학습 가능한 파라미터 수 출력


        self.mlm_head = nn.Linear(self.llada_hidden_size, self.rvq_n_emb, dtype=model_dtype)

    def forward(self, masked_global_eeg_ids_for_input, attention_mask=None, mlm_labels=None):
        model_outputs = self.llada_model(
            input_ids=masked_global_eeg_ids_for_input,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 중간 은닉 상태들을 출력하도록 요청
            return_dict=True
        )

        # output_hidden_states=True로 설정하면, model_outputs.hidden_states 에 모든 레이어의 은닉 상태가 튜플 형태로 저장됩니다.
        # 이 튜플의 마지막 요소가 우리가 원하는 last_hidden_state 입니다.
        # (입력 임베딩 결과 + 각 트랜스포머 레이어의 출력 결과)
        all_hidden_states = model_outputs.hidden_states
        sequence_output = all_hidden_states[-1] # 마지막 트랜스포머 레이어의 출력

        # --- 디버깅을 위한 print 문 (여전히 유효합니다) --- #
        #print(f"Shape of sequence_output (from hidden_states[-1]) before mlm_head: {sequence_output.shape}")
        # 이제 예상되는 모양: (batch_size, sequence_length, llada_hidden_size), 예: (1, 64, 4096)

        mlm_logits = self.mlm_head(sequence_output)

        loss = None
        if mlm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(mlm_logits.view(-1, self.rvq_n_emb), mlm_labels.view(-1))

        return {
            "loss": loss,
            "logits": mlm_logits,
            # "hidden_states": sequence_output # 필요하다면 전체 hidden_states 튜플을 반환할 수도 있습니다.
        }

def evaluate_model(model, dataloader, device, rvq_tokenizer):
    model.eval()  # 모델을 평가 모드로 설정
    total_val_loss = 0
    
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for batch_eeg_tensors in dataloader:
            batch_eeg_tensors = batch_eeg_tensors.to(device)

            # 1. RVQ 토큰화
            _, local_eeg_indices_batch = rvq_tokenizer(batch_eeg_tensors)
            original_local_eeg_ids = local_eeg_indices_batch.squeeze(1)
            
            # 2. 글로벌 ID 변환 및 마스킹
            global_original_eeg_ids = original_local_eeg_ids + model.v_text
#             global_original_eeg_ids = original_local_eeg_ids + model.eeg_token_offset
            masked_global_eeg_ids_for_input, masked_indices = forward_process_eeg(
                global_original_eeg_ids, 
                model.global_mask_eeg_token_id
            )

            # 3. MLM 레이블 생성
            mlm_labels = original_local_eeg_ids.clone()
            mlm_labels[~masked_indices] = -100

            # 4. 어텐션 마스크 생성
            attention_mask = torch.ones_like(original_local_eeg_ids, dtype=torch.float32, device=device)

#             # ------------------- ① 토큰 id 범위 검사 ------------------- #
#             vocab_sz = model.llada_model.get_input_embeddings().weight.size(0)
#             if masked_global_eeg_ids_for_input.max() >= vocab_sz:
#                 bad_ids = masked_global_eeg_ids_for_input[
#                     masked_global_eeg_ids_for_input >= vocab_sz]
#                 raise RuntimeError(
#                     f"[BUG] found out-of-range ids {bad_ids.tolist()}  (vocab={vocab_sz})")
#             # ---------------------------------------------------------- #

            # 5. 모델 순전파 및 손실 계산
            outputs = model(
                masked_global_eeg_ids_for_input=masked_global_eeg_ids_for_input,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels
            )
            loss = outputs["loss"]
            if loss is not None:
                total_val_loss += loss.item()
            else:
                print("검증 중 손실이 None입니다.") # 이 경우는 거의 없어야 함

    avg_val_loss = total_val_loss / len(dataloader)
    model.train() # 모델을 다시 학습 모드로 설정
    return avg_val_loss

def collate_fn_eeg_mlm(batch):
    eeg_tensors = [item[0] for item in batch] # item[0]이 eeg_tensor라고 가정
    return torch.stack(eeg_tensors)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLADA_MODEL_NAME = "GSAI-ML/LLaDA-8B-Base" # 또는 사용 중인 모델명
RVQ_N_EMB = 512  # RVQ 코드북의 임베딩 개수 (어휘 크기)
RVQ_N_Q = 12     # RVQ 코드북 개수 (토큰 시퀀스 길이)
BATCH_SIZE = 64   # GPU 메모리에 맞게 조정
INIT_LR = 1e-4   # 초기 학습률
NUM_EPOCHS = 10   # 학습 에폭 수
VALIDATION_SPLIT = 0.1
max_grad_norm = 1.0
model_save_path_base = "/home/work/skku/hyo/hyo/model/eeg_llada_mlm_model"
grid_search_results = []

GRID_SEARCH_BASE_DIR = "/home/work/skku/hyo/hyo/grid_search_results" # 모든 그리드 서치 결과 저장 기본 폴더
os.makedirs(GRID_SEARCH_BASE_DIR, exist_ok=True)
OVERALL_BEST_MODEL_DIR = os.path.join(GRID_SEARCH_BASE_DIR, "overall_best_model")
if os.path.exists(OVERALL_BEST_MODEL_DIR):
    shutil.rmtree(OVERALL_BEST_MODEL_DIR)
os.makedirs(OVERALL_BEST_MODEL_DIR, exist_ok=True)

print(f"사용 디바이스: {DEVICE}")

rvq_tokenizer = RVQTokenizer(
    feat=840, 
    latent=128, # RVQ 내부 임베딩 차원, LLaDA hidden size와 다름
    n_q=RVQ_N_Q, 
    n_emb=RVQ_N_EMB, 
    hidden=256,
    TOKENIZER_CHECKPOINT_PATH="/home/work/skku/hyo/hyo/model/rvq_best_model_sen_512.pt" # 실제 경로로 수정!
).to(DEVICE)
rvq_tokenizer.eval() # 토크나이저는 학습하지 않으므로 eval 모드

eeg_dataset_full = EEGDataset(data_dir="/home/work/skku/hyo/hyo/dataset/sentence.parquet") # 실제 경로로 수정!

param_grid = {
    'learning_rate': [1e-4],
    'lora_r': [8, 16, 32],
    'lora_alpha': [16, 32, 64],
    'batch_size': [64], # GPU 메모리 상황에 따라 조절
    'validation_split' : [0.1]
}

keys, values = zip(*param_grid.items())
hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"총 {len(hyperparameter_combinations)}개의 하이퍼파라미터 조합으로 그리드 서치를 수행합니다.")

all_epoch_logs_list = [] # 모든 에폭 로그를 저장할 리스트
overall_best_val_loss = float('inf')

# 그리드 서치 조합들에 대한 tqdm 프로그레스 바
combo_pbar = tqdm(enumerate(hyperparameter_combinations), 
                  total=len(hyperparameter_combinations),
                  desc="Grid Search Progress",
                  unit="combo")

for combo_idx, params in combo_pbar:
    combo_id_str = f"combo_{combo_idx+1:03d}_lr_{params['learning_rate']}_r_{params['lora_r']}_alpha_{params['lora_alpha']}_bs_{params['batch_size']}"
    
    # tqdm 설명 업데이트
    combo_pbar.set_description(f"Combo {combo_idx+1}/{len(hyperparameter_combinations)}")
    combo_pbar.set_postfix({
        'lr': f"{params['learning_rate']:.2e}",
        'r': params['lora_r'],
        'alpha': params['lora_alpha'],
        'bs': params['batch_size']
    })
    
    print(f"\n--- 그리드 서치 조합 {combo_idx+1}/{len(hyperparameter_combinations)} ({combo_id_str}) 시작 ---")
    print(f"현재 하이퍼파라미터: {params}")

    current_combo_save_dir = os.path.join(GRID_SEARCH_BASE_DIR, combo_id_str)
    os.makedirs(current_combo_save_dir, exist_ok=True)

    current_lr = params['learning_rate']
    current_lora_r = params['lora_r']
    current_lora_alpha = params['lora_alpha']
    current_batch_size = params['batch_size']
    current_validation_slplit = params['validation_split']

    set_seeds(SEED) 

    # 데이터셋 분할
    dataset_size = len(eeg_dataset_full)
    val_size = int(dataset_size * current_validation_slplit)
    train_size = dataset_size - val_size
    
    print(f"전체 데이터셋 크기: {dataset_size}")
    print(f"학습 데이터셋 크기: {train_size}")
    print(f"검증 데이터셋 크기: {val_size}")
    
    # random_split을 사용하여 데이터셋 분할 (시드 고정으로 재현성 확보 가능)
    # torch.manual_seed(42) # 필요시 시드 고정
    train_dataset, val_dataset = random_split(eeg_dataset_full, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn_eeg_mlm, worker_init_fn=None)
    val_dataloader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collate_fn_eeg_mlm, worker_init_fn=None)
    
    print(f"학습 데이터로더 크기: {len(train_dataloader)}")
    print(f"검증 데이터로더 크기: {len(val_dataloader)}")
    
    # EEG_LLaDA_MLM 모델 초기화 (이전 코드에서 정의된 클래스 사용)
    qlora_params_config = {
        "r": current_lora_r,
        "lora_alpha": current_lora_alpha,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"] # LLaDA 모델 구조에 맞게 명시적 지정 권장 (이전 안내 참조)
    }
    model = EEG_LLaDA_MLM(
        llada_model_name=LLADA_MODEL_NAME, 
        rvq_n_emb=RVQ_N_EMB, 
        use_qlora=True,
        qlora_config_params=qlora_params_config
    ).to(DEVICE)

    params_to_optimize = []
    print("\n옵티마이저를 위한 파라미터 수집 중:")
    for name, param in model.llada_model.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            #print(f"  LLaDA (PEFT): {name} (모양: {param.shape}, dtype: {param.dtype})")
    
    for name, param in model.mlm_head.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            #print(f"  MLM 헤드: {name} (모양: {param.shape}, dtype: {param.dtype})")
    
    if not params_to_optimize:
        raise ValueError("학습할 파라미터가 없습니다. 모델 설정을 확인하세요.")
    print(f"옵티마이저를 위한 총 파라미터 그룹 수: {len(params_to_optimize)}")
    
    optimizer = optim.AdamW(params_to_optimize, lr=current_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print("\n옵티마이저 및 스케줄러 설정 완료.")

    best_val_loss_this_combo = float('inf')

    # 에폭 진행을 위한 tqdm 프로그레스 바
    epoch_pbar = tqdm(range(NUM_EPOCHS), 
                      desc=f"Epochs (Combo {combo_idx+1})",
                      leave=False,
                      unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        total_train_loss_epoch = 0
        epoch_start_time = time.time()

        # 학습 배치들에 대한 tqdm 프로그레스 바
        train_pbar = tqdm(train_dataloader, 
                         desc=f"Training Epoch {epoch+1}",
                         leave=False,
                         unit="batch")
                         
        for step, batch_eeg_tensors in enumerate(train_pbar):
            batch_eeg_tensors = batch_eeg_tensors.to(DEVICE)
            with torch.no_grad():
                _, local_eeg_indices_batch = rvq_tokenizer(batch_eeg_tensors)
                original_local_eeg_ids = local_eeg_indices_batch.squeeze(1)
            global_original_eeg_ids = original_local_eeg_ids + model.v_text
            masked_global_eeg_ids_for_input, masked_indices = forward_process_eeg(
                global_original_eeg_ids, model.global_mask_eeg_token_id)
            mlm_labels = original_local_eeg_ids.clone()
            mlm_labels[~masked_indices] = -100
            attention_mask = torch.ones_like(original_local_eeg_ids, dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            outputs = model(masked_global_eeg_ids_for_input=masked_global_eeg_ids_for_input,
                            attention_mask=attention_mask, mlm_labels=mlm_labels)
            loss = outputs["loss"]
            if loss is None: continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
            optimizer.step()
            total_train_loss_epoch += loss.item()

            # 현재까지의 평균 손실을 표시
            current_avg_loss = total_train_loss_epoch / (step + 1)
            train_pbar.set_postfix({'loss': f'{current_avg_loss:.4f}'})
        
        train_pbar.close()
        
        avg_train_loss_epoch = total_train_loss_epoch / len(train_dataloader)

        print(f"  검증 중...")
        avg_val_loss_epoch = evaluate_model(model, val_dataloader, DEVICE, rvq_tokenizer) # evaluate_model 함수는 이전에 정의됨

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # 에폭 프로그레스 바 업데이트
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss_epoch:.4f}',
            'val_loss': f'{avg_val_loss_epoch:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'time': f'{epoch_duration:.1f}s'
        })
        
        print(f"  조합 {combo_idx+1}, 에폭 {epoch+1}: Train Loss={avg_train_loss_epoch:.4f}, Val Loss={avg_val_loss_epoch:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(avg_val_loss_epoch)

        # CSV 로깅을 위한 데이터 추가
        epoch_log_entry = params.copy() # 현재 하이퍼파라미터 복사
        epoch_log_entry['combo_id_str'] = combo_id_str
        epoch_log_entry['combo_idx'] = combo_idx + 1
        epoch_log_entry['epoch'] = epoch + 1
        epoch_log_entry['train_loss'] = avg_train_loss_epoch
        epoch_log_entry['validation_loss'] = avg_val_loss_epoch
        epoch_log_entry['current_lr_epoch_end'] = optimizer.param_groups[0]['lr']
        all_epoch_logs_list.append(epoch_log_entry)

        # 매 에폭 모델 저장
        epoch_model_save_dir = os.path.join(current_combo_save_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_model_save_dir, exist_ok=True)
        model.llada_model.save_pretrained(os.path.join(epoch_model_save_dir, "qlora_adapter"))
        torch.save(model.mlm_head.state_dict(), os.path.join(epoch_model_save_dir, "mlm_head.pth"))
        print(f"    에폭 {epoch+1} 모델 저장 완료: {epoch_model_save_dir}")

        # 현재 조합 내에서 베스트 모델 업데이트 및 저장
        if avg_val_loss_epoch < best_val_loss_this_combo:
            best_val_loss_this_combo = avg_val_loss_epoch
            combo_best_model_save_dir = os.path.join(current_combo_save_dir, "best_model_in_combo")
            os.makedirs(combo_best_model_save_dir, exist_ok=True)
            model.llada_model.save_pretrained(os.path.join(combo_best_model_save_dir, "qlora_adapter"))
            torch.save(model.mlm_head.state_dict(), os.path.join(combo_best_model_save_dir, "mlm_head.pth"))
            print(f"    조합 내 베스트 모델 갱신 (에폭 {epoch+1}), Val Loss: {best_val_loss_this_combo:.4f}. 저장 완료: {combo_best_model_save_dir}")

        # 전체 그리드 서치 중 베스트 모델 업데이트 및 저장
        if avg_val_loss_epoch < overall_best_val_loss:
            overall_best_val_loss = avg_val_loss_epoch
            print(f"    ✨ 전체 베스트 모델 갱신 (조합 {combo_idx+1}, 에폭 {epoch+1}), Val Loss: {overall_best_val_loss:.4f}. 저장 중...")
            # if os.path.exists(OVERALL_BEST_MODEL_DIR): # 이전 베스트 모델 폴더 삭제
            #     shutil.rmtree(OVERALL_BEST_MODEL_DIR)
            # os.makedirs(OVERALL_BEST_MODEL_DIR, exist_ok=True) # 삭제 후 다시 생성
            model.llada_model.save_pretrained(os.path.join(OVERALL_BEST_MODEL_DIR, "qlora_adapter"))
            torch.save(model.mlm_head.state_dict(), os.path.join(OVERALL_BEST_MODEL_DIR, "mlm_head.pth"))
            # 베스트 모델 정보 저장 (어떤 조합과 에폭이었는지)
            with open(os.path.join(OVERALL_BEST_MODEL_DIR, "best_model_info.txt"), "w") as f:
                f.write(f"Best model from combination: {combo_id_str}\n")
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Validation Loss: {overall_best_val_loss:.4f}\n")
                f.write(f"Hyperparameters: {params}\n")
            print(f"    ✨ 전체 베스트 모델 저장 완료: {OVERALL_BEST_MODEL_DIR}")
            
    epoch_pbar.close()
    print(f"--- 그리드 서치 조합 {combo_idx+1} 완료. 이 조합의 최저 검증 손실: {best_val_loss_this_combo:.4f} ---")

    # --- 메모리 해제 시작 ---
    print(f"조합 {combo_idx+1}에 사용된 객체들의 메모리 해제를 시도합니다...")
    # 1. 모델, 옵티마이저, 스케줄러 삭제
    del model
    del optimizer
    del scheduler
    # 필요하다면 데이터로더도 삭제 (만약 루프 내에서 매번 재생성된다면)
    # del train_dataloader
    # del val_dataloader 
    # (주의: train_dataset, val_dataset은 random_split으로 생성되므로, 
    #  eeg_dataset_full이 루프 밖에 있다면 이들은 삭제하지 않아도 됩니다.
    #  만약 eeg_dataset_full도 루프 안에서 매번 로드한다면 삭제 대상입니다.)

    # 2. GPU 캐시 비우기 (PyTorch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU 캐시를 비웠습니다.")

    # 3. 파이썬 가비지 컬렉터 명시적 호출
    collected_count = gc.collect()
    print(f"가비지 컬렉터가 {collected_count}개의 객체를 수거했습니다.")
    # --- 메모리 해제 완료 ---

combo_pbar.close()
print(f"\n🎉 전체 그리드 서치 완료! 최종 베스트 검증 손실: {overall_best_val_loss:.4f}")
