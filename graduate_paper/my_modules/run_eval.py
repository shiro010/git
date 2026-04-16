import unicodedata
import re
import string
import os
from datetime import datetime
import json
from rank_bm25 import BM25Okapi

from get_model_safe import get_model_safe
from main_inference import main_inference
from ScalingController import ScalingController


IS_SAVE = True
LOG_DIR = "/content/drive/MyDrive/Research_Logs"

class ExperimentLogger:
    def __init__(self, log_dir=LOG_DIR, exp_name="experiment", dataset="hotpot", beta=1.0, scaling_method="distribution"):
        # ログ保存用フォルダ作成
        if IS_SAVE:
            os.makedirs(log_dir, exist_ok=True)

            # ファイル名に日時を入れる (上書き防止)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filepath = os.path.join(log_dir, f"{exp_name}_{dataset}_{str(beta)}_{scaling_method}_{timestamp}.jsonl")

            print(f"📝 Logging to: {self.filepath}")

    def log_sample(self, data_dict):
        """1件分のデータを保存"""
        if IS_SAVE:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")

# ==========================================
# 1. 評価指標 (Exact Match) のための関数群
# ==========================================

def normalize_answer(s):
    """
    回答の正規化を行う関数。
    これを通すことで、「The Apple.」と「apple」を同一とみなします。
    """
    s = unicodedata.normalize("NFKC", s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction, ground_truth):
    """単純な文字列一致判定（正規化後）"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    return normalized_prediction == normalized_ground_truth

def calculate_em(prediction, ground_truths):
    """
    正解リスト(ground_truths)のうち、どれか1つでも一致すれば1.0(正解)とする
    """
    # HotpotQAのように正解が文字列1つだけの場合もリスト化して共通処理
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    # 全ての正解候補に対してEMを計算し、最大値(1 or 0)を取る
    scores = [exact_match_score(prediction, gt) for gt in ground_truths]
    return max(scores)

def calculate_substring_score(prediction, ground_truths):
    """
    部分一致評価 (Substring Match / Answer Recall)
    予測文の中に、正解の文字列が含まれていれば 1.0 (正解) とする。
    例:
      Pred: "The answer is Apple."
      Gold: "Apple"
      -> "apple" in "the answer is apple" -> True (正解)
    """
    # 正規化（小文字化など）
    pred_norm = normalize_answer(prediction)

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    # リスト内のどれか1つでも予測文の中に含まれていればOK
    # g (正解) が p (予測) の中に in しているかチェック
    matches = [(normalize_answer(gt) in pred_norm) for gt in ground_truths]

    return 1.0 if any(matches) else 0.0

# ==========================================
# 2. 評価実行ループ
# ==========================================

def run_eval_exact_match(dataset, model_path, num_samples=100, beta=1.0, scaling_method="distribution"):

    logger = ExperimentLogger(dataset="hotpot", beta=beta, scaling_method=scaling_method)

    # 設定を最初の行に記録しておくと便利
    meta_info = {
        "type": "META",
        "model": model_path,
        "beta": beta,
        "scaling_method": scaling_method,
        "num_samples": num_samples
    }
    logger.log_sample(meta_info)

    # モデルの準備
    model, tokenizer = get_model_safe(model_path)

    # 統計用変数
    total_em = 0
    count = 0

    print(f"🚀 Exact Match Evaluation Start: Processing {num_samples} samples")
    print("-" * 60)

    for i in range(num_samples):
        try:
            item = dataset[i]
            query = item["question"]
            gold_answer = item["answer"] # HotpotQAはここが文字列、NQはリストの場合あり

            # --- 文書の整形 & BM25リランキング ---
            titles = item["context"]["title"]
            sentences_list = item["context"]["sentences"]

            raw_documents = []
            tokenized_docs = []
            for title, sents in zip(titles, sentences_list):
                full_doc = f"{title}\n{''.join(sents)}"
                raw_documents.append(full_doc)
                tokenized_docs.append(full_doc.split())

            # BM25スコア計算
            bm25 = BM25Okapi(tokenized_docs)
            raw_scores = [float(s) for s in bm25.get_scores(query.split())]

            # スコア順に並べ替え (降順)
            combined = sorted(zip(raw_scores, raw_documents), key=lambda x: x[0], reverse=True)
            sorted_scores, sorted_documents = zip(*combined)
            sorted_scores = list(sorted_scores)
            sorted_documents = list(sorted_documents)

            controller = ScalingController(method=scaling_method, beta=beta)
            gamma_list = controller.compute_gammas(sorted_scores) # 各文書ごとのγ


            # --- 推論実行 ---
            generated_answer = main_inference(
                model=model,
                tokenizer=tokenizer,
                documents=sorted_documents,
                gamma_list=gamma_list,
                query=query,
                scaling_method=scaling_method,
                beta=beta
            )

            # --- ★Exact Match 計算★ ---
            # ここで 1.0 (正解) か 0.0 (不正解) が返る
            em_score = calculate_substring_score(generated_answer, gold_answer)

            total_em += em_score
            count += 1

            # 結果表示 (正解なら ✅, 不正解なら ❌)
            mark = "✅" if em_score == 1.0 else "❌"
            print(f"Q{i+1}: {query}")
            print(f"Gen: {generated_answer}")
            print(f"Ref: {gold_answer}")
            print(f"Result: {mark} (EM: {int(em_score)})")
            print("-" * 30)

            # ログ保存
            log_data = {
                "id": i,
                "question": query,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "em_score": em_score,
                # 分析用に重要なデータ
                "doc_scores": [round(s, 2) for s in sorted_scores[:5]], # 上位5件のスコア
                "top_doc_title": sorted_documents[0].split("\n")[0] if sorted_documents else ""
            }
            logger.log_sample(log_data)

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"❌ Failed to load model:\n{error_msg}")
            return


    # --- 最終スコア ---
    if count > 0:
        final_score = 100.0 * total_em / count
        print("=" * 60)
        print(f"📊 Final Exact Match (EM): {final_score:.2f}%")
        print("=" * 60)
    else:
        print("No samples processed.")