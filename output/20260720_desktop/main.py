import os

# XLA JIT を無効化（autotuner コンパイル失敗回避）。TensorFlow を import する前に設定する必要がある
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import gc

import sys
# プロジェクトルートを import パスに追加（lib/ をパッケージとして解決）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.preprocess import load_data, load_model, save_org_data, save_ae_data, DATA_DIR
from lib.differential_evolution_varlen import (
    generate_population_varlen,
    differential_evolution_varlen,
)
from keras.utils import to_categorical

# TensorFlowのGPU設定
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

INIT_POPULATION_SIZE = 300
PRED_BATCH_SIZE = 128

DE_ALPHA = 0.8

def _predict_batch(model, x_np, batch_size=PRED_BATCH_SIZE):
    """
    複数候補をまとめて順伝播する GPU 一括推論。x_np: (S, T, 1) -> (S, nb_classes)。
    model.predict は XLA コンパイル経路を通り autotuner で失敗し得るため eager 呼び出し model(...) を使う。
    """
    x_np = np.asarray(x_np, dtype=np.float32)
    probs_all = []
    for start in range(0, x_np.shape[0], batch_size):
        chunk = x_np[start:start + batch_size]
        prep = model(chunk, training=False)
        probs_all.append(np.asarray(prep))
    return np.concatenate(probs_all, axis=0)

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"利用可能なGPU: {len(gpus)}個")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        except RuntimeError as e:
            print(f"GPU設定エラー: {e}")
    else:
        print("GPUが見つかりません。CPUで実行します。")

GLOBAL_START_DATE = None
OUTPUT_BASE_DIR = "output"

def create_output_dirs(date_str):
    date_dir = os.path.join(OUTPUT_BASE_DIR, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)

    plot_dir = os.path.join(date_dir, "plot")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    gen_plot_dir = os.path.join(date_dir, "generation_plot")
    if not os.path.exists(gen_plot_dir):
        os.makedirs(gen_plot_dir)

    return date_dir, plot_dir, gen_plot_dir

def backup_code(target_dir):
    current_file = os.path.abspath(__file__)
    filename = "main.py"
    dst_path = os.path.join(target_dir, filename)
    try:
        shutil.copy2(current_file, dst_path)
        print(f"コード記述のバックアップを作成しました: {dst_path}")
    except Exception as e:
        print(f"バックアップ作成エラー: {e}")

def align_to_length(series_array: np.ndarray, target_length: int) -> np.ndarray:
    array_1d = np.asarray(series_array, dtype=np.float32).reshape(-1)
    current_length = array_1d.shape[0]
    if target_length is None or current_length == target_length:
        return array_1d
    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(array_1d, (0, pad_width), mode='edge')
    return array_1d[:target_length]


def create_comparison_plot(dataset, model_type, strategy, func, sel_mode, index, org_data, ae_data, model, current_date, lim, true_label=None, output_dir=None):
    try:
        org_row = org_data.iloc[index].values
        ae_row = ae_data.iloc[index].values
        if len(org_row) == 0 or len(ae_row) == 0:
            return

        try:
            try:
                input_length = int(getattr(model, 'input_shape', (None, None, 1))[1])
            except Exception:
                input_length = None

            org_aligned = align_to_length(org_row, input_length)
            ae_aligned = align_to_length(ae_row, input_length)

            org_processed = org_aligned.reshape(1, -1, 1).astype(np.float32)
            ae_processed = ae_aligned.reshape(1, -1, 1).astype(np.float32)

            org_pred = model(org_processed, training=False).numpy()
            ae_pred = model(ae_processed, training=False).numpy()

            org_class = np.argmax(org_pred[0])
            ae_class = np.argmax(ae_pred[0])
            if org_class == ae_class:
                classification_result = "same"
                result_color = "green"
            else:
                classification_result = "different"
                result_color = "red"

            if true_label is not None:
                true_label_int = int(true_label)
                ae_correct = (ae_class == true_label_int)
                true_label_result = f"True: {true_label_int}, Org pred: {org_class}, AE pred: {ae_class} ({'correct' if ae_correct else 'misclassified'})"
                true_label_color = "green" if ae_correct else "red"
            else:
                true_label_result = None
        except Exception as e:
            print(f"分類エラー (Sample {index+1}): {str(e)}")
            classification_result = "error"
            result_color = "gray"
            true_label_result = None

        plt.figure(figsize=(12, 6))
        x_org = np.arange(1, len(org_row) + 1)
        x_ae = np.arange(1, len(ae_row) + 1)
        plt.plot(x_org, org_row, color='blue', linewidth=2, label='Original', alpha=0.8)
        plt.plot(x_ae, ae_row, color='orange', linewidth=2, label='Perturbed', alpha=0.8)
        plt.title(f'{dataset} - {model_type.upper()} - {strategy} - {func} - {sel_mode} - Sample {index+1}', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.text(0.02, 0.98, f'Org vs AE pred: {classification_result}',
                 transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                 color=result_color, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if true_label_result is not None:
            plt.text(0.02, 0.90, true_label_result,
                     transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                     color=true_label_color, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_dir:
            base_plot_dir = os.path.join(output_dir, "plot")
        else:
            if not os.path.exists('plot'):
                os.makedirs('plot')
            base_plot_dir = 'plot'

        dataset_dir = os.path.join(base_plot_dir, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        filename = f'{current_date}_{dataset}_{model_type}_{strategy}_{func}_{sel_mode}_lim{lim}_{index+1}.png'
        filepath = os.path.join(dataset_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"画像保存完了: {filepath}")
    except Exception as e:
        print(f"画像生成エラー (Sample {index+1}): {str(e)}")
        plt.close()

def save_result_to_excel(
    filename: str,
    dataset_name: str,
    model: str,
    eval_func_name: str,
    strategy: str,
    selection_mode: str,
    original_accuracy: float,
    attack_accuracy: float,
    start_time: datetime,
    end_time: datetime,
    gen: int,
    mae: float,
    mse: float,
    rmse: float,
    lim: float,
    avg_num_perturbations: float = None,
    avg_amp: float = None,
    max_perturbations: int = None,
    output_dir: str = None
):
    duration = (end_time - start_time).total_seconds()
    minutes = int(duration / 60)
    result_row = {
        "開始日時": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "終了日時": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "実行時間（秒）": duration,
        "実行時間（分）": minutes,
        "平均世代数": gen,
        "モデル名": model,
        "データセット名": dataset_name,
        "戦略": strategy,
        "選択モード": selection_mode,
        "Lim": lim,
        "最大摂動数": max_perturbations,
        "平均摂動数": avg_num_perturbations,
        "平均振幅(1摂動あたり)": avg_amp,
        "評価関数": eval_func_name,
        "元精度": original_accuracy,
        "攻撃後精度": attack_accuracy,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

    if output_dir:
        filename = os.path.join(output_dir, "results_lab.xlsx")

    file_exists = os.path.exists(filename)
    if file_exists:
        df = pd.read_excel(filename)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    df.to_excel(filename, index=False)

def save_detailed_results(detailed_results, output_dir, dataset, model_type, strategy, func, sel_mode, lim):
    if not detailed_results:
        return

    filename = f"detailed_log_{dataset}_{model_type}_{strategy}_{func}_{sel_mode}_lim{lim}.xlsx"
    filepath = os.path.join(output_dir, filename)

    df = pd.DataFrame(detailed_results)
    df.to_excel(filepath, index=False)
    print(f"詳細ログ保存完了: {filepath}")

def save_generation_log(generation_logs, output_dir, dataset, model_type, strategy, func, sel_mode, lim):
    if not generation_logs:
        return

    filename = f"generation_log_{dataset}_{model_type}_{strategy}_{func}_{sel_mode}_lim{lim}.csv"
    filepath = os.path.join(output_dir, filename)

    df = pd.DataFrame(generation_logs)
    df.to_csv(filepath, index=False)
    print(f"世代ログ保存完了: {filepath}")

def plot_generation_history(generation_logs, output_dir, dataset, model_type, strategy, func, sel_mode, lim, maxiter=50):
    if not generation_logs:
        return

    sample_logs = {}
    for log in generation_logs:
        sample_id = log["sample_id"]
        if sample_id not in sample_logs:
            sample_logs[sample_id] = []
        sample_logs[sample_id].append(log)

    for sample_id, logs in sample_logs.items():
        logs.sort(key=lambda x: x["generation"])

        generations = [x["generation"] for x in logs]
        values = [x["best_value"] for x in logs]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, values, linestyle='-', color='b')

        plt.title(f'Sample {sample_id}\n{dataset} {model_type} {strategy} {func} {sel_mode}')
        plt.xlabel('Generation')
        plt.ylabel('Confidence')
        plt.ylim(0, 1.05)
        plt.xlim(0, maxiter)
        plt.grid(True, linestyle='--', alpha=0.7)

        filename = f"generation_history_{dataset}_{model_type}_{strategy}_{func}_{sel_mode}_lim{lim}_{sample_id}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"世代推移プロット保存完了: {output_dir}")

# ========= 戦略（ラベル用） =========
strategies = [
    "varlen_rand1exp"
]


def apply_pairs(x_sample, pairs):
    """個体 [[pos, amp], ...] を適用（位置重複は先勝ち）。戻り値: (適用後系列, 摂動数 k)。"""
    x_mod = np.copy(x_sample)
    tlen = x_mod.shape[0]
    used_positions = set()
    for pos, amp in pairs:
        p = int(round(pos))
        p = max(0, min(p, tlen - 1))
        if p in used_positions:
            continue
        used_positions.add(p)
        x_mod[p] = x_mod[p] + amp
    return x_mod, len(pairs)


def make_fitness_batch(model, x_sample, label, alpha, K_max, lim):
    """集団全体を1バッチで評価する関数を返す（GPU一括推論を保つ）。
    評価値 = alpha*低下量 + beta*摂動数 + gamma*摂動振幅（すべて [-1,0]、小さいほど良い）。"""
    T = x_sample.shape[0]
    org = x_sample.reshape(1, T, 1).astype(np.float32)

    def fitness_batch(population):
        alpha = 0.7
        beta = 0.1
        gamma = 0.2
        S = len(population)
        adv = np.repeat(org, S, axis=0)     # (S, T, 1)
        ks = np.empty(S, dtype=np.float64)
        amp_sum = np.zeros(S, dtype=np.float64)
        for s, ind in enumerate(population):
            ks[s] = len(ind)
            used_positions = set()
            for pos, amp in ind:
                amp_sum[s] += abs(amp)
                p = int(round(pos))
                p = max(0, min(p, T - 1))
                if p in used_positions:     # 重複は先勝ち
                    continue
                used_positions.add(p)
                adv[s, p] = adv[s, p] + amp
        probs = _predict_batch(model, adv)              # (S, nb_classes)
        misclass = -(1.0 - probs[:, label])             # [-1,0] 低下量
        npert = -(1.0 - ks / max(1, K_max))             # [-1,0] 摂動数
        amp_ratio = np.clip(amp_sum / (lim * np.maximum(ks, 1)), 0.0, 1.0) 
        val_amp = -(1.0 - amp_ratio)                    # [-1,0] 摂動振幅（小さいほど良い）
        return alpha * misclass + beta * npert + gamma * val_amp  # [-1,0]

    return fitness_batch


class GenerationLogger:
    """可変長DEの世代ログ。best_value=1-conf（純粋な誤分類度）、num_perturbations=長さk。
    駆動ループから callback(gen, best_ind, best_fit) で呼ばれる。"""
    def __init__(self, model, x, label, history):
        self.model = model
        self.x = x
        self.label = label
        self.history = history
        self.generation = 0

    def __call__(self, gen, best_ind, best_fit):
        self.generation += 1
        x_mod, k = apply_pairs(self.x, best_ind)
        probs = self.model(x_mod.reshape(1, -1, 1)).numpy()[0]
        self.history.append({
            "generation": self.generation,
            "best_value": 1.0 - float(probs[self.label]),
            "num_perturbations": k
        })


def screen_initial_candidates(model, x, y_true, amp=0.3, top_m=5):
    """初期個体改良（main.py 準拠）。
    各位置に ±amp の単点摂動を入れて1バッチ推論し、真クラス確信度を最も下げる符号の
    スコアを位置ごとに取る。スコア = -(1 - conf)（小さいほど攻撃が効く）の良い順に上位 top_m 位置を返す。
    旧版と違い ±両符号を試し、score>0 のフィルタもしない（常に top_m 位置を種にする）。"""
    x1d = np.asarray(x, dtype=np.float32).reshape(-1)         # (T,)
    T = x1d.shape[0]
    advs = np.repeat(x1d.reshape(1, T), 2 * T, axis=0)        # (2T, T)
    idx = np.arange(T)
    advs[2 * idx, idx] += amp                                  # 各位置に +amp
    advs[2 * idx + 1, idx] -= amp                              # 各位置に -amp
    probs = _predict_batch(model, advs.reshape(2 * T, T, 1))   # (2T, nb_classes)
    scores = -(1.0 - probs[:, y_true])                         # (2T,) [-1,0] 小さいほど良い
    best_per_pos = np.minimum(scores[0::2], scores[1::2])      # 各位置で ± の良い方 (T,)
    order = np.argsort(best_per_pos)                           # 昇順（良い順）
    return [(float(best_per_pos[p]), int(p)) for p in order[:top_m]]


def process_single_sample(index, x_sample, y_sample, model,
                          K_max, lim, max_generations, strategy, tlen, alpha):
    """単一サンプルの処理（自前可変長DE・GPU一括推論）。
    個体は可変長リスト [[pos,amp],...]。変異=可変長rand1（x1基底＋±1ペア）、交叉=exp、選択=greedy。
    評価は集団を1バッチで GPU 推論（make_fitness_batch）。"""
    label = int(np.argmax(y_sample))

    sample_for_screen = np.copy(x_sample)
    while sample_for_screen.ndim > 2 and sample_for_screen.shape[-1] == 1:
        sample_for_screen = np.squeeze(sample_for_screen, axis=-1)

    base_positions = []
    try:
        top_candidates = screen_initial_candidates(
            model=model, x=sample_for_screen, y_true=label, amp=lim, top_m=K_max
        )
        base_positions = [pos for _, pos in top_candidates]
    except Exception as e:
        print(f"初期候補生成に失敗 (Sample {index}): {e}")

    generation_logs = []
    fitness_batch = make_fitness_batch(model, x_sample, label, alpha, K_max, lim)
    population = generate_population_varlen(INIT_POPULATION_SIZE, K_max, tlen, lim, base_positions)
    logger = GenerationLogger(model, x_sample, label, generation_logs)

    best_ind, best_fit, nit = differential_evolution_varlen(
        fitness_batch=fitness_batch,
        population=population,
        K_max=K_max, 
        tlen=tlen, 
        lim=lim,
        generations=max_generations,
        F_range=(0.5, 1.0), 
        CR=0.7,
        callback=logger, 
        seed=42
    )

    x_modified, optimized_num_perturbations = apply_pairs(x_sample, best_ind)
    adv_x = x_modified.reshape(1, tlen, 1)
    probs = model(adv_x).numpy()[0]
    success = (int(np.argmax(probs)) != label)
    conf = float(probs[label])

    detailed_info = {
        "sample_id": index,
        "success": success,
        "message": "",
        "fun": best_fit,
        "nit": nit,
        "nfev": (nit + 1) * INIT_POPULATION_SIZE,
        "confidence": conf,
        "optimized_num_perturbations": optimized_num_perturbations
    }

    for log_item in generation_logs:
        log_item["sample_id"] = index

    status = "成功" if success else "失敗"
    print(f"サンプル {index + 1} の処理完了 ({status}, 採用 k={optimized_num_perturbations}, conf={conf:.3f}, 世代={nit})")

    return {
        "index": index,
        "x_modified": x_modified,
        "detailed_info": detailed_info,
        "generation_logs": generation_logs,
        "nit": nit
    }

def main(dataset, model_type, lim=0.1):
    for strategy in strategies:
        for func in ["varlen_rand1"]:
            for sel_mode in ["varlen"]:

                current_date_dir, current_plot_dir, current_gen_plot_dir = create_output_dirs(GLOBAL_START_DATE)

                gen = []
                detailed_results = []
                all_generation_logs = []

                x_test, y_test = load_data(dataset, is_test=True)

                nb_classes = len(np.unique(y_test))
                y_true = y_test
                y_test = to_categorical(y_test)
                tlen = len(x_test[0])

                max_generations = 200
                K_max = 15
                alpha = DE_ALPHA
                print(f"\n[{dataset}] 可変長DE(自前lib): K_max={K_max}, alpha={alpha}, データ長: {tlen}")

                model = load_model(dataset, model_type)

                save_org_data(dataset, model_type, x_test)

                pred = model(x_test)
                test_acc = tf.metrics.SparseCategoricalAccuracy()
                test_acc(y_true, pred)
                org_acc = test_acc.result().numpy() * 100

                start = datetime.datetime.now()
                print(f"GPU一括推論版（自前可変長DE）開始: サンプル数={len(x_test)}, 初期個体数={INIT_POPULATION_SIZE}")

                # サンプルは直列ループ、各DEは集団を1バッチ評価（model を共有）
                optimized_np_list = []
                for index in range(len(x_test)):
                    result = process_single_sample(
                        index, x_test[index], y_test[index], model,
                        K_max, lim, max_generations, strategy, tlen, alpha
                    )
                    x_test[index] = result["x_modified"]
                    detailed_results.append(result["detailed_info"])
                    all_generation_logs.extend(result["generation_logs"])
                    gen.append(result["nit"])
                    optimized_np_list.append(result["detailed_info"]["optimized_num_perturbations"])

                    if (index + 1) % 10 == 0:
                        print(f"  進捗: {index + 1}/{len(x_test)} 完了")

                print(f"処理完了: {len(x_test)}サンプル処理済み")

                end = datetime.datetime.now()

                save_ae_data(dataset, model_type, x_test)

                pred = model(x_test)
                test_acc = tf.metrics.SparseCategoricalAccuracy()
                test_acc(y_true, pred)
                ae_acc = test_acc.result().numpy() * 100

                df_true = pd.read_csv(os.path.join(DATA_DIR, dataset, f"{dataset}_{model_type}_TEST_ORG.tsv"), sep="\t", header=None)
                df_pred = pd.read_csv(os.path.join(DATA_DIR, dataset, f"{dataset}_{model_type}_TEST_AE.tsv"), sep="\t", header=None)

                assert df_true.shape == df_pred.shape, "データの形状が一致していません"

                y_true_flat = df_true.values.flatten()
                y_pred_flat = df_pred.values.flatten()

                mae = mean_absolute_error(y_true_flat, y_pred_flat)   # 従来のMAE（全セル平均）
                mse = mean_squared_error(y_true_flat, y_pred_flat)
                rmse = np.sqrt(mse)
                # 1摂動あたりの平均振幅：摂動が加わったセル（|元-摂動後|>0）だけで平均する。
                # 非摂動セルは差0なので、全セル平均のMAEと違い「1個の摂動がどれくらいの量か」を表す。
                abs_diff = np.abs(y_true_flat - y_pred_flat)
                perturbed = abs_diff[abs_diff > 1e-12]
                avg_amp = float(perturbed.mean()) if perturbed.size > 0 else 0.0

                avg_gen = np.mean(gen) if len(gen) > 0 else 0
                avg_np = np.mean(optimized_np_list) if len(optimized_np_list) > 0 else 0

                save_result_to_excel(
                    filename="results.xlsx",
                    dataset_name=dataset,
                    model=model_type,
                    eval_func_name=func,
                    strategy=strategy,
                    selection_mode=sel_mode,
                    original_accuracy=org_acc,
                    attack_accuracy=ae_acc,
                    start_time=start,
                    end_time=end,
                    gen=avg_gen,
                    mae=mae,
                    mse=mse,
                    rmse=rmse,
                    lim=lim,
                    avg_num_perturbations=avg_np,
                    avg_amp=avg_amp,
                    max_perturbations=K_max,
                    output_dir=current_date_dir
                )

                save_detailed_results(detailed_results, current_date_dir, dataset, model_type, strategy, func, sel_mode, lim)
                save_generation_log(all_generation_logs, current_date_dir, dataset, model_type, strategy, func, sel_mode, lim)
                plot_generation_history(all_generation_logs, current_gen_plot_dir, dataset, model_type, strategy, func, sel_mode, lim, maxiter=max_generations)

                print(f"画像生成開始: {dataset} - {model_type} - {strategy} - {func} - {sel_mode}")
                try:
                    org_data = pd.read_csv(os.path.join(DATA_DIR, dataset, f"{dataset}_{model_type}_TEST_ORG.tsv"), sep="\t", header=None)
                    ae_data = pd.read_csv(os.path.join(DATA_DIR, dataset, f"{dataset}_{model_type}_TEST_AE.tsv"), sep="\t", header=None)

                    org_labels = org_data.iloc[:, 0]
                    org_series = org_data.iloc[:, 1:]
                    ae_series = ae_data.iloc[:, 1:]

                    num_rows = min(len(org_series), len(ae_series))
                    print(f"画像生成対象データ行数: {num_rows}")

                    for i in range(num_rows):
                        create_comparison_plot(
                            dataset=dataset,
                            model_type=model_type,
                            strategy=strategy,
                            func=func,
                            sel_mode=sel_mode,
                            index=i,
                            org_data=org_series,
                            ae_data=ae_series,
                            model=model,
                            current_date=GLOBAL_START_DATE,
                            lim=lim,
                            true_label=org_labels.iloc[i],
                            output_dir=current_date_dir
                        )
                        if (i + 1) % 10 == 0:
                            print(f"画像生成進捗: {i + 1}/{num_rows} 完了")

                    print(f"画像生成完了: {dataset} - {model_type} - {strategy} - {func} - {sel_mode} ({num_rows}枚)")

                except Exception as e:
                    print(f"画像生成エラー ({dataset}_{model_type}_{strategy}_{func}_{sel_mode}): {str(e)}")

                del model
                gc.collect()

def get_unique_dir_name(base_dir, date_str):
    original_path = os.path.join(base_dir, date_str)
    if not os.path.exists(original_path):
        return date_str

    counter = 1
    while True:
        new_date_str = f"{date_str}_{counter}"
        new_path = os.path.join(base_dir, new_date_str)
        if not os.path.exists(new_path):
            return new_date_str
        counter += 1

if __name__ == "__main__":
    # GPU設定を初期化（単一プロセスで1枚のGPUを使用）
    configure_gpu()

    base_date_str = datetime.datetime.now().strftime("%Y%m%d")
    GLOBAL_START_DATE = get_unique_dir_name(OUTPUT_BASE_DIR, base_date_str)

    d_dir, _, _ = create_output_dirs(GLOBAL_START_DATE)
    backup_code(d_dir)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU一括推論版（自前可変長DE / lib.differential_evolution_varlen）を使用: GPU数={len(gpus)}")
    print(f"初期個体数: {INIT_POPULATION_SIZE} (固定)")
    print(f"可変長DE: 個体=[[pos,amp],...]の可変長。変異=rand1(x1基底＋±1ペア)+exp交叉、選択=greedy。評価=alpha*低下量+(1-alpha)*摂動数")

    start_all = datetime.datetime.now()

    lim = 0.3

    print(f"\n========== Start Experiment lim={lim}, max_perturbations=15, init_pop={INIT_POPULATION_SIZE} ==========")
    main("BeetleFly", "fcn", lim=lim)
    main("Car", "fcn", lim=lim)
    main("Coffee", "fcn", lim=lim)
    main("Computers", "fcn", lim=lim)
    main("ECG200", "fcn", lim=lim)
    main("ShapeletSim", "fcn", lim=lim)
    main("ToeSegmentation2", "fcn", lim=lim)

    main("BeetleFly", "resnet", lim=lim)
    main("Car", "resnet", lim=lim)
    main("Coffee", "resnet", lim=lim)
    main("Computers", "resnet", lim=lim)
    main("ECG200", "resnet", lim=lim)
    main("ShapeletSim", "resnet", lim=lim)
    main("ToeSegmentation2", "resnet", lim=lim)

    end_all = datetime.datetime.now()
    print(f"開始日時：{start_all}")
    print(f"終了日時：{end_all}")