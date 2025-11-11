import csv, json, time, torch, numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── 모델·토크나이저·데이터로더는 노트북에서 저장된 checkpoint / pickle 등을 로드 ───
from my_project import (
    model,                    # 학습해 둔 LLaDA 모델
    test_dataloader,          # DataLoader
    unified_eeg_text_tokenizer,
    generate,
    calculate_bleu_scores,
    calculate_rouge_scores,
    calculate_wer,
)

# ─── 샘플링 하이퍼파라미터 ───
GEN_STEPS   = 64
GEN_LEN     = 128
BLOCK_LEN   = 64
TEMP        = 0.3
CFG_SCALE   = 0.9
REMASKING   = "low_confidence"
MASK_ID     = 126336                    # 필요 시

# ─── 결과 저장용 ───
out_dir      = Path("/home/work/skku/hyo/generate")
out_dir.mkdir(exist_ok=True)
csv_path     = out_dir / "eeg_gen_results.csv"
metrics_path = out_dir / "final_metrics.json"

def main():
    device = next(model.parameters()).device
    model.eval()

    # CSV 헤더
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reference", "hypothesis"])

    all_hyp, all_ref = [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="🔮  Generating from EEG"):
            eeg_list = batch["batched_eeg_data"]
            ref_list = batch["batched_reference_texts"]

            for eeg_tensor, ref_text in zip(eeg_list, ref_list):
                # 1) build prompt
                prompt_pack = unified_eeg_text_tokenizer.build_chat_template_prompt(eeg_tensor)
                prompt_ids  = prompt_pack["input_ids"].to(device)
                prompt_len  = prompt_pack["prompt_len"]

                # 2) generate
                out_ids = generate(
                    model,
                    prompt      = prompt_ids,
                    steps       = GEN_STEPS,
                    gen_length  = GEN_LEN,
                    block_length= BLOCK_LEN,
                    temperature = TEMP,
                    cfg_scale   = CFG_SCALE,
                    remasking   = REMASKING,
                )

                # 3) decode
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

if __name__ == "__main__":
    main()
