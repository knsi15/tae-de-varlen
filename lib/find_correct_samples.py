"""
FCN / ResNet が正解しているテストサンプルの一覧を抽出するスクリプト。

main_lab.py で全テストデータに対して差分進化を実行すると時間がかかるため、
まず「モデルが正しく分類できているサンプル」だけを対象にランダムサンプリングして
実行数を絞り込むための前処理として使う。

使い方:
    python -m lib.find_correct_samples
    python -m lib.find_correct_samples --datasets BeetleFly Car --sample-size 10 --seed 42
    python -m lib.find_correct_samples --all
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import tensorflow as tf

# Windows のコンソール(cp932)は例外メッセージ中の一部Unicode文字を
# エンコードできずクラッシュすることがあるため、置換出力にしておく
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.preprocess import DATA_DIR, MODELS_DIR, load_data, load_model

DEFAULT_DATASETS = [
    "BeetleFly",
    "Car",
    "Coffee",
    "Computers",
    "ECG200",
    "ShapeletSim",
    "ToeSegmentation2",
]
DEFAULT_MODEL_TYPES = ["fcn", "resnet"]


def discover_all_datasets(model_types):
    # data ディレクトリと、指定した全モデル種のディレクトリの両方に
    # フォルダが存在するデータセットのみを対象にする
    model_sets = None
    for model_type in model_types:
        model_root = os.path.join(MODELS_DIR, model_type, "UCRArchive_2018_itr_8")
        names = set(os.listdir(model_root))
        model_sets = names if model_sets is None else (model_sets & names)

    data_sets = set(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    return sorted(model_sets & data_sets)


def _predict_batch(model, x_np, batch_size=128):
    # StarLightCurves のような大規模データセットは一括推論すると
    # 中間テンソルが巨大になりOOMするため、バッチに分けて推論する
    probs_all = []
    for start in range(0, x_np.shape[0], batch_size):
        chunk = x_np[start:start + batch_size]
        probs_all.append(np.asarray(model(chunk, training=False)))
    return np.concatenate(probs_all, axis=0)


def find_correct_indices(dataset, model_type):
    x_test, y_test = load_data(dataset, is_test=True)
    model = load_model(dataset, model_type)
    try:
        probs = _predict_batch(model, x_test)
    finally:
        # load_model() のたびにグラフ/変数が積み上がりメモリを食い潰すため、
        # 使い終わったモデルは都度解放する（85データセット×2モデルを連続実行するため必須）
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    y_pred = np.argmax(probs, axis=1)
    y_true = y_test.astype(int)
    correct_mask = (y_pred == y_true)
    correct_indices = np.where(correct_mask)[0].tolist()
    return {
        "dataset": dataset,
        "model_type": model_type,
        "num_total": int(len(y_true)),
        "num_correct": int(correct_mask.sum()),
        "accuracy": float(correct_mask.mean()),
        "correct_indices": correct_indices,
    }


def sample_indices(correct_indices, sample_size, seed):
    rng = np.random.RandomState(seed)
    if sample_size is None or sample_size >= len(correct_indices):
        return sorted(correct_indices)
    return sorted(rng.choice(correct_indices, size=sample_size, replace=False).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--model-types", nargs="+", default=DEFAULT_MODEL_TYPES)
    parser.add_argument("--all", action="store_true",
                         help="data/models 両方に存在する全データセットを対象にする")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join("output", "correct_samples.json"))
    args = parser.parse_args()

    if args.all:
        datasets = discover_all_datasets(args.model_types)
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = DEFAULT_DATASETS

    print(f"対象データセット数: {len(datasets)}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {}
    errors = []
    for di, dataset in enumerate(datasets, 1):
        for model_type in args.model_types:
            try:
                info = find_correct_indices(dataset, model_type)
            except Exception as e:
                print(f"[{dataset}/{model_type}] エラーのためスキップ: {e}")
                errors.append({"dataset": dataset, "model_type": model_type, "error": str(e)})
                continue
            info["sampled_indices"] = sample_indices(
                info["correct_indices"], args.sample_size, args.seed
            )
            results.setdefault(dataset, {})[model_type] = info
            print(
                f"({di}/{len(datasets)}) [{dataset}/{model_type}] "
                f"正解 {info['num_correct']}/{info['num_total']} "
                f"(acc={info['accuracy']*100:.2f}%) "
                f"-> サンプリング {len(info['sampled_indices'])}件: {info['sampled_indices']}"
            )
            # 途中で落ちても結果を失わないよう、データセット完了ごとに保存する
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n保存完了: {args.output}")
    if errors:
        print(f"エラー件数: {len(errors)}")
        for err in errors:
            print(f"  {err['dataset']}/{err['model_type']}: {err['error']}")


if __name__ == "__main__":
    main()
