import torch

def main_inference(
    model,
    tokenizer,
    documents, # List[str]: 検索された文書の内容
    gamma_list,    # List[float]: 文書の検索スコア
    query,     # str: 質問
    scaling_method="distribution", # "probabilistic" or "distribution"
    beta=1.0
):
    # プロンプト構築とトークン位置の特定
    full_prompt = "Answer the question based on the following context.\n"
    doc_spans = [] # 各文書がトークン列のどこにあるか (start, end) を記録

    current_text = full_prompt

    # 文書を追加しながら位置を記録
    for i, doc_content in enumerate(documents):
        header = f"[Document {i+1}]: "

        prefix_tokens = tokenizer.encode(current_text, add_special_tokens=True)
        start_idx = len(prefix_tokens)

        doc_text = f"{header}{doc_content}\n"
        current_text += doc_text

        current_tokens = tokenizer.encode(current_text, add_special_tokens=True)
        end_idx = len(current_tokens)

        doc_spans.append({
            "doc_id": i,
            "start_idx": start_idx,
            "end_idx": end_idx
        })

    # クエリ追加
    current_text += f"\nQuestion: {query}\nAnswer:"
    # print(current_text)
    input_ids = tokenizer(current_text, return_tensors="pt").input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    # 4. Scale Map の作成 [1, seq_len]
    # デフォルトは 1.0 (何もしない)
    scale_map = torch.ones((1, seq_len), device=model.device, dtype=torch.float16)

    print(f"Applying scaling: {scaling_method}")
    for span, gamma in zip(doc_spans, gamma_list):
        start = span["start_idx"]
        end = span["end_idx"]
        # その文書に対応するトークン区間を γ で埋める
        if start < seq_len and end <= seq_len:
            scale_map[0, start:end] = gamma

    # モデルに scale_map をセット
    model.config.current_scale_map = scale_map

    # 5. 生成実行
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.0, # 論文設定
            do_sample=False
        )

    seq_len = input_ids.shape[1]

    generated_text = tokenizer.decode(output[0][seq_len:], skip_special_tokens=True)

    del output

    return generated_text.strip()