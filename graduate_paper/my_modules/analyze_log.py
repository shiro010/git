# analyze files
import json

def load_jsonl_robust(file_path):
    """
    JSONLファイルを読み込み、IDをキーとした辞書を返す。
    1行目などのメタデータ（'id'を持たない行）はデータとして扱わず、別途返す。
    """
    data_map = {}
    metadata = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: JSON decode error at line {i+1} in {file_path}")
                continue

            # 'id' キーがある場合はデータ行とみなす
            if "id" in item:
                data_map[item['id']] = item

            # 'type': 'META' または 'id' がない場合はメタデータとみなす
            elif i == 0 or item.get("type") == "META":
                metadata = item

    return metadata, data_map

def compare_results(file_path1, file_path2):
    # データをロード
    meta1, data1 = load_jsonl_robust(file_path1)
    meta2, data2 = load_jsonl_robust(file_path2)

    # メタ情報の表示（設定確認用）
    print(f"=== File 1 Info: {file_path1} ===")
    if meta1:
        print(f"  Method: {meta1.get('scaling_method', 'N/A')}, Beta: {meta1.get('beta', 'N/A')}")
    else:
        print("  (No Metadata found)")

    print(f"\n=== File 2 Info: {file_path2} ===")
    if meta2:
        print(f"  Method: {meta2.get('scaling_method', 'N/A')}, Beta: {meta2.get('beta', 'N/A')}")
    else:
        print("  (No Metadata found)")
    print("-" * 40)

    # 共通するIDを取得
    common_ids = set(data1.keys()) & set(data2.keys())

    stats = {
        "same": 0,           # 完全一致
        "diff": 0,           # 違う
        "improved": 0,       # 0.0 -> 1.0 (改善)
        "worsened": 0,       # 1.0 -> 0.0 (悪化)
    }

    # 差分ログ用
    changed_examples = []

    print(f"比較対象データ数 (ID一致): {len(common_ids)}件")

    for doc_id in common_ids:
        score1 = data1[doc_id].get('em_score', 0.0)
        score2 = data2[doc_id].get('em_score', 0.0)

        if score1 == score2:
            stats["same"] += 1
        else:
            stats["diff"] += 1

            if score1 < score2:
                stats["improved"] += 1
            else:
                stats["worsened"] += 1

            # 最初の5件だけ詳細を保存
            if len(changed_examples) < 5:
                changed_examples.append({
                    "id": doc_id,
                    "score_A": score1,
                    "score_B": score2,
                    "question": data1[doc_id].get("question", "")
                })

    # === 結果出力 ===
    print("\n【集計結果】")
    print(f"  EMスコアが同じ : {stats['same']} 件")
    print(f"  EMスコアが違う : {stats['diff']} 件")
    print(f"    - 改善 (A < B) : {stats['improved']} 件")
    print(f"    - 悪化 (A > B) : {stats['worsened']} 件")

    if changed_examples:
        print("\n【変化したデータの例 (先頭5件)】")
        for item in changed_examples:
            print(f"  ID: {item['id']} | {item['score_A']} -> {item['score_B']} | Q: {item['question']}")

import json
import os
import statistics
from tabulate import tabulate # %pip install tabulate しておくと綺麗に表示されます

class ExperimentAnalyzer:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def summarize_file(self, filename):
        """特定のログファイルの結果を集計する"""
        filepath = os.path.join(self.log_dir, filename)
        results = []
        meta = {}

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # META情報の取得
                if data.get("type") == "META":
                    meta = data
                    continue
                # スコアの抽出 (em_score または calculate_em 等のキーに対応)
                score = data.get("em_score")
                if score is not None:
                    results.append(score)

        if not results:
            return None

        avg_score = statistics.mean(results) * 100
        return {
            "filename": filename,
            "avg_em": avg_score,
            "count": len(results),
            "meta": meta
        }

    def report_all_logs(self):
        """ログディレクトリ内の全ファイルをスキャンして表形式で表示する"""
        summary_list = []
        files = [f for f in os.listdir(self.log_dir) if f.endswith(".jsonl")]

        for f in sorted(files):
            res = self.summarize_file(f)
            if res:
                summary_list.append([
                    res["filename"],
                    f"{res['avg_em']:.2f}%",
                    res["count"],
                    res["meta"].get("beta", "-"),
                    res["meta"].get("scaling_method", "-")
                ])

        headers = ["File Name", "Avg EM", "Samples", "Beta", "Method"]
        print("\n" + "="*80)
        print("📊 Experiment Summary Report")
        print("="*80)
        print(tabulate(summary_list, headers=headers, tablefmt="grid"))