"""
main_verBatch_with_init.py - 初期個体あり版
スクリーニングで有望な摂動位置を特定し、初期個体群として DE に渡す。
遺伝子選択モード: front / back / random / lim
"""

import numpy as np
import tensorflow as tf
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import gc
import multiprocessing
from multiprocessing import Pool, cpu_count

import sys
# プロジェクトルート（scripts/ の親）を import パスに追加し lib/ を解決
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.preprocess import load_data, load_model, save_org_data, save_ae_data
from keras.utils import to_categorical

# TensorFlowのGPU設定
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

# 初期個体数（固定）
INIT_POPULATION_SIZE = 300

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
    filename = os.path.basename(current_file)
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


# ========= 評価関数 =========

def eval_misclassification(probs, perturbations, label, original_x, perturbed_x, lim):
    return -(1 - probs[0, label])

def eval_combined_normalized(probs, perturbations, label, original_x, perturbed_x, lim):
    alpha = 0.6
    misclass_score = -(1 - probs[0, label])
    mae = np.mean(np.abs(perturbed_x - original_x))
    perturb_score = -max(0.0, 1 - (mae / lim))
    return alpha * misclass_score + (1-alpha) * perturb_score

def eval_combined_with_npert(probs, perturbations, label, original_x, perturbed_x, lim):
    alpha = 0.4
    beta  = 0.3
    gamma = 0.3
    misclass_score = -(1 - probs[0, label])
    mae = np.mean(np.abs(perturbed_x - original_x))
    perturb_score = -max(0.0, 1 - (mae / lim))
    num_active = int(round(perturbations[0]))
    max_perturbations = (len(perturbations) - 1) // 2
    npert_score = -max(0.0, 1 - (num_active / max_perturbations))
    return alpha * misclass_score + beta * perturb_score + gamma * npert_score

def eval_misclass_with_npert(probs, perturbations, label, original_x, perturbed_x, lim):
    alpha = 0.6
    misclass_score = -(1 - probs[0, label])
    num_active = int(round(perturbations[0]))
    max_perturbations = (len(perturbations) - 1) // 2
    npert_score = -max(0.0, 1 - (num_active / max_perturbations))
    return alpha * misclass_score + (1-alpha) * npert_score

eval_funcs = {
    "misclass_with_npert": eval_misclass_with_npert,
}

strategies = [
    "rand1exp"
]

# 遺伝子選択モード: 最適化された摂動数 num_active 個をどのように選ぶか
selection_modes = ["front", "back", "random", "lim"]


# ========= 遺伝子選択ヘルパー =========

def select_perturbations(pair_genes, num_active, mode):
    """
    pair_genes: (pos, amp) のフラット配列 (長さ max_perturbations * 2)
    num_active: 使用する摂動数
    mode: "front" | "back" | "random" | "lim"
    戻り値: 選ばれた (pos, amp) ペアのリスト
    """
    all_pairs = [pair_genes[i:i+2] for i in range(0, len(pair_genes), 2)]
    n = len(all_pairs)
    num_active = max(1, min(num_active, n))

    if mode == "front":
        return all_pairs[:num_active]
    elif mode == "back":
        return all_pairs[n - num_active:]
    elif mode == "random":
        indices = np.random.choice(n, num_active, replace=False)
        return [all_pairs[i] for i in sorted(indices)]
    elif mode == "lim":
        # 摂動値の絶対値が小さい順（変化量が小さい順）に選ぶ
        sorted_pairs = sorted(all_pairs, key=lambda p: abs(p[1]))
        return sorted_pairs[:num_active]
    else:
        raise ValueError(f"Unknown selection mode: {mode}")


# ========= 目的関数 =========

def objective_function_multiitem(perturbations, model, x, y, nb_classes, eval_func, max_perturbations, lim, selection_mode):
    """
    perturbations[0]: 有効な摂動数 (実数→整数に丸める)
    perturbations[1:]: (pos, amp) のペアが max_perturbations 個
    """
    num_active = int(round(perturbations[0]))
    num_active = max(1, min(num_active, max_perturbations))

    label = np.argmax(y)
    x = np.copy(x)

    org_x = np.copy(x).reshape(1, x.shape[0], x.shape[1])
    adv_x = x.reshape(1, x.shape[0], x.shape[1])

    pair_genes = perturbations[1:]
    plist = select_perturbations(pair_genes, num_active, selection_mode)

    used_positions = set()
    for p in plist:
        pos = int(p[0])
        if pos in used_positions:
            continue
        used_positions.add(pos)
        val = p[1]
        adv_x[0, pos] = adv_x[0, pos] + val

    prep = model(adv_x)

    return eval_func(prep.numpy(), perturbations, label, org_x, adv_x, lim)


class GenerationLogger:
    def __init__(self, func, args):
        self.func = func
        self.args = args
        self.history = []
        self.generation = 0

    def __call__(self, xk, convergence=None):
        self.generation += 1
        val = self.func(xk, *self.args)

        saved_val = -val
        num_active = int(round(xk[0]))
        num_active = max(1, min(num_active, self.args[5]))  # args[5] = max_perturbations
        self.history.append({
            "generation": self.generation,
            "best_value": saved_val,
            "num_perturbations": num_active
        })


# ========= 初期解指定 =========

def evaluate_confidence(model, x, y_true):
    probs = model.predict(x[np.newaxis, :], verbose=0)[0]
    return probs[y_true]

def add_sparse_perturbation(x, pos, amp):
    x_new = x.copy()
    x_new[pos] += amp
    return x_new

def screen_initial_candidates(model, x, y_true, amp=0.3, top_m=5):
    L = len(x)
    original_conf = evaluate_confidence(model, x, y_true)

    candidates = []
    for pos in range(L):
        x_pert = add_sparse_perturbation(x, pos, amp)
        conf = evaluate_confidence(model, x_pert, y_true)
        score = original_conf - conf
        candidates.append((score, pos))

    candidates = [(s, p) for s, p in candidates if s > 0]
    candidates.sort(reverse=True, key=lambda t: t[0])
    return candidates[:top_m]

def make_initial_population(top_candidates, dim, tlen, lim, max_perturbations):
    """
    SciPy DE に渡すための初期個体行列 (INIT_POPULATION_SIZE, dim) を作る。
    dim = 1 (有効摂動数) + max_perturbations * 2 (pos, amp のペア)
    """
    S = INIT_POPULATION_SIZE
    init_pop = np.zeros((S, dim))

    base_candidate_positions = [pos for _, pos in top_candidates]
    init_num_perturbations = max(1, len(base_candidate_positions))

    for i in range(S):
        init_pop[i, 0] = init_num_perturbations + np.random.uniform(-1, 1)

        candidate_positions = list(base_candidate_positions)
        while len(candidate_positions) < max_perturbations:
            candidate_positions.append(np.random.randint(0, tlen))
        np.random.shuffle(candidate_positions)

        for j in range(max_perturbations):
            pos_idx = 1 + j * 2
            amp_idx = 1 + j * 2 + 1
            init_pop[i, pos_idx] = candidate_positions[j]
            init_pop[i, amp_idx] = np.random.uniform(-lim * 0.3, lim * 0.3)

    return init_pop

def process_single_sample(index, x_sample, y_sample, dataset, model_type, perturbation,
                          nb_classes, eval_func, max_perturbations, lim, max_generations,
                          strategy, tlen, selection_mode):
    """
    単一サンプルの処理を行う関数（並列処理用）
    """
    configure_gpu()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    print(f"サンプル {index + 1} の処理を開始... (選択モード: {selection_mode})")

    model = load_model(dataset, model_type)

    current_args = (model, x_sample, y_sample, nb_classes, eval_func, max_perturbations, lim, selection_mode)
    logger = GenerationLogger(objective_function_multiitem, current_args)

    sample_for_screen = np.copy(x_sample)
    while sample_for_screen.ndim > 2 and sample_for_screen.shape[-1] == 1:
        sample_for_screen = np.squeeze(sample_for_screen, axis=-1)

    sample_label = np.argmax(y_sample)

    top_candidates = []
    try:
        top_candidates = screen_initial_candidates(
            model=model,
            x=sample_for_screen,
            y_true=sample_label,
            amp=lim,
            top_m=max_perturbations
        )
    except Exception as e:
        print(f"初期候補生成に失敗 (Sample {index}): {e}")

    dim = len(perturbation)

    init_pop = make_initial_population(
        top_candidates,
        dim,
        tlen,
        lim,
        max_perturbations
    )

    result = differential_evolution(
        func=objective_function_multiitem,
        bounds=perturbation,
        args=current_args,
        strategy=strategy,
        maxiter=max_generations,
        popsize=15,   # init配列を渡す場合はpopsizeは無視される
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=False,
        seed=42,
        callback=logger,
        init=init_pop,
        tol=0,
        atol=-1
    )

    best_perturbations = result.x

    optimized_num_perturbations = int(round(best_perturbations[0]))
    optimized_num_perturbations = max(1, min(optimized_num_perturbations, max_perturbations))

    detailed_info = {
        "sample_id": index,
        "success": getattr(result, "success", False),
        "message": getattr(result, "message", ""),
        "fun": result.fun,
        "nit": result.nit,
        "nfev": result.nfev,
        "optimized_num_perturbations": optimized_num_perturbations
    }

    generation_logs = []
    for log_item in logger.history:
        log_item_copy = log_item.copy()
        log_item_copy["sample_id"] = index
        generation_logs.append(log_item_copy)

    # 最良個体の摂動を適用
    x_modified = np.copy(x_sample)
    pair_genes = best_perturbations[1:]
    plist = select_perturbations(pair_genes, optimized_num_perturbations, selection_mode)

    used_positions = set()
    for p in plist:
        pos = int(p[0])
        if pos in used_positions:
            continue
        used_positions.add(pos)
        val = p[1]
        x_modified[int(pos)][0][0] = x_modified[int(pos)][0][0] + val

    del model
    gc.collect()

    print(f"サンプル {index + 1} の処理完了 (世代数: {result.nit})")

    return {
        "index": index,
        "x_modified": x_modified,
        "detailed_info": detailed_info,
        "generation_logs": generation_logs,
        "nit": result.nit
    }


# ========= メイン =========

def main(dataset, model_type, lim=0.1, num_workers=None):
    for strategy in strategies:
        for func in eval_funcs:
            for sel_mode in selection_modes:

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
                max_perturbations = 15
                print(f"\n[{dataset}] 選択モード: {sel_mode}, 最大摂動数: {max_perturbations}, データ長: {tlen}")

                base = [(0, tlen - 1), (-lim, lim)]
                perturbation = [(1, max_perturbations)] + base * max_perturbations

                model = load_model(dataset, model_type)

                save_org_data(dataset, model_type, x_test)

                pred = model(x_test)
                test_acc = tf.metrics.SparseCategoricalAccuracy()
                test_acc(y_true, pred)
                org_acc = test_acc.result().numpy() * 100

                start = datetime.datetime.now()
                print(f"並列処理開始: CPU数={cpu_count()}, サンプル数={len(x_test)}, 初期個体数={INIT_POPULATION_SIZE}")

                process_args = []
                for index in range(len(x_test)):
                    process_args.append((
                        index, x_test[index], y_test[index], dataset, model_type, perturbation,
                        nb_classes, eval_funcs[func], max_perturbations, lim, max_generations,
                        strategy, tlen, sel_mode
                    ))

                gpus = tf.config.list_physical_devices('GPU')
                has_gpu = len(gpus) > 0

                if num_workers is None:
                    if has_gpu:
                        num_workers_actual = min(12, len(x_test))
                        print(f"GPU検出: ワーカー数を{num_workers_actual}に制限します")
                    else:
                        num_workers_actual = min(cpu_count(), len(x_test))
                else:
                    num_workers_actual = min(num_workers, len(x_test))

                print(f"使用ワーカー数: {num_workers_actual}")
                with Pool(processes=num_workers_actual) as pool:
                    results = pool.starmap(process_single_sample, process_args)

                optimized_np_list = []
                for result in results:
                    index = result["index"]
                    x_test[index] = result["x_modified"]
                    detailed_results.append(result["detailed_info"])
                    all_generation_logs.extend(result["generation_logs"])
                    gen.append(result["nit"])
                    optimized_np_list.append(result["detailed_info"]["optimized_num_perturbations"])

                print(f"並列処理完了: {len(results)}サンプル処理済み")

                end = datetime.datetime.now()

                save_ae_data(dataset, model_type, x_test)

                pred = model(x_test)
                test_acc = tf.metrics.SparseCategoricalAccuracy()
                test_acc(y_true, pred)
                ae_acc = test_acc.result().numpy() * 100

                df_true = pd.read_csv(f"data/{dataset}/{dataset}_{model_type}_TEST_ORG.tsv", sep="\t", header=None)
                df_pred = pd.read_csv(f"data/{dataset}/{dataset}_{model_type}_TEST_AE.tsv", sep="\t", header=None)

                assert df_true.shape == df_pred.shape, "データの形状が一致していません"

                y_true_flat = df_true.values.flatten()
                y_pred_flat = df_pred.values.flatten()

                mae = mean_absolute_error(y_true_flat, y_pred_flat)
                mse = mean_squared_error(y_true_flat, y_pred_flat)
                rmse = np.sqrt(mse)

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
                    max_perturbations=max_perturbations,
                    output_dir=current_date_dir
                )

                save_detailed_results(detailed_results, current_date_dir, dataset, model_type, strategy, func, sel_mode, lim)
                save_generation_log(all_generation_logs, current_date_dir, dataset, model_type, strategy, func, sel_mode, lim)
                plot_generation_history(all_generation_logs, current_gen_plot_dir, dataset, model_type, strategy, func, sel_mode, lim, maxiter=max_generations)

                print(f"画像生成開始: {dataset} - {model_type} - {strategy} - {func} - {sel_mode}")
                try:
                    org_data = pd.read_csv(f"data/{dataset}/{dataset}_{model_type}_TEST_ORG.tsv", sep="\t", header=None)
                    ae_data = pd.read_csv(f"data/{dataset}/{dataset}_{model_type}_TEST_AE.tsv", sep="\t", header=None)

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
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    configure_gpu()

    base_date_str = datetime.datetime.now().strftime("%Y%m%d")
    GLOBAL_START_DATE = get_unique_dir_name(OUTPUT_BASE_DIR, base_date_str)

    d_dir, _, _ = create_output_dirs(GLOBAL_START_DATE)
    backup_code(d_dir)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"バッチ処理版（初期個体あり）を使用: CPU数={cpu_count()}, GPU数={len(gpus)}")
    print(f"初期個体数: {INIT_POPULATION_SIZE} (固定)")
    print(f"選択モード: {selection_modes}")

    start_all = datetime.datetime.now()

    lim = 0.3

    print(f"\n========== Start Experiment lim={lim}, max_perturbations=15, init_pop={INIT_POPULATION_SIZE} ==========")
    main("BeetleFly", "fcn", lim=lim)
    main("Car", "fcn", lim=lim)
    main("Coffee", "fcn", lim=lim)
    main("Computers", "fcn", lim=lim)
    main("ECG200", "fcn", lim=lim)
    main("ToeSegmentation2", "fcn", lim=lim)
    main("ShapeletSim", "fcn", lim=lim)

    end_all = datetime.datetime.now()
    print(f"開始日時：{start_all}")
    print(f"終了日時：{end_all}")
