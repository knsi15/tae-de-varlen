import os

# XLA JIT を無効化（autotuner コンパイル失敗回避）。TensorFlow を import する前に設定する必要がある
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import numpy as np
import tensorflow as tf
from scipy.optimize import differential_evolution
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
from keras.utils import to_categorical

# TensorFlowのGPU設定
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

INIT_POPULATION_SIZE = 300
PRED_BATCH_SIZE = 128

DE_ALPHA = 0.6         # [0,1] 評価値 = alpha*低下量 + (1-alpha)*摂動数（ともに[-1,0]）。大きいほど攻撃成功を優先

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

# ========= 戦略（ラベル用。実変異はカスタム strategy callable） =========
strategies = [
    "varlen_rand1"
]

# ========= 可変長エンコーディングのヘルパー =========
# 個体は固定長ベクトル: genome[0]=長さk, 続けて K_max 個の (pos, amp) ペア。
# 活性は先頭 k ペア。残りはパディング（評価時に長さ k でゲートされ無視される）。

def _rint(rng, low, high):
    """RandomState(.randint) と Generator(.integers) の両対応で [low, high) の整数を返す。
    scipy は seed=int のとき rng に RandomState を渡すため、.integers は使えない。"""
    if rng is None:
        return int(np.random.randint(low, high))
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def genome_to_pairs(genome, K_max):
    """先頭の長さ遺伝子 k に従い、先頭 k 個の [pos, amp] を返す（活性集合）。"""
    k = int(round(genome[0]))
    k = max(1, min(k, K_max))
    return [[genome[1 + 2 * j], genome[1 + 2 * j + 1]] for j in range(k)]


def pairs_to_genome(pairs, K_max, tlen, lim, rng=None):
    """ペア列を固定長ベクトル (1 + 2*K_max,) に書き戻す。
    先頭=長さ、活性スロットは pos/amp をクリップ、残りは pos ランダム・amp=0 でパディング。"""
    g = np.empty(1 + 2 * K_max, dtype=np.float64)
    k = max(1, min(len(pairs), K_max))
    g[0] = k
    for j in range(K_max):
        if j < k:
            g[1 + 2 * j] = min(max(pairs[j][0], 0), tlen - 1)
            g[1 + 2 * j + 1] = min(max(pairs[j][1], -lim), lim)
        else:
            g[1 + 2 * j] = _rint(rng, 0, tlen)
            g[1 + 2 * j + 1] = 0.0
    return g


def extend_pairs_zero(pairs, n, tlen, rng):
    """短い→長い: amp=0 のペアを末尾に足して長さ n にする（差分整列用、pos はランダム）。"""
    out = [list(p) for p in pairs]
    while len(out) < n:
        out.append([float(_rint(rng, 0, tlen)), 0.0])
    return out


def shrink_pairs_by_amp(pairs, n):
    """長い→短い: |amp| の小さい順に並べ替え、先頭 n 個を残す（小振幅を残す方針）。"""
    order = sorted(range(len(pairs)), key=lambda i: abs(pairs[i][1]))
    return [list(pairs[i]) for i in order[:max(1, n)]]


def align_pairs_zero(pairs, n, tlen, rng):
    """差分の整列: 短ければ amp=0 で伸長、長ければ |amp| 小さい順で短縮。"""
    if len(pairs) < n:
        return extend_pairs_zero(pairs, n, tlen, rng)
    if len(pairs) > n:
        return shrink_pairs_by_amp(pairs, n)
    return [list(p) for p in pairs]


def add_random_pair(pairs, tlen, lim, rng):
    """長さ+1 の突然変異: 実効振幅を持つ新規 (pos, amp) ペアを末尾に追加。"""
    out = [list(p) for p in pairs]
    out.append([float(_rint(rng, 0, tlen)), float(rng.uniform(-lim, lim))])
    return out


def make_varlen_strategy(K_max, tlen, lim, F_range=(0.5, 1.0)):
    """scipy differential_evolution 用カスタム strategy callable（可変長 rand1）。
    v = x1 + F*(x2 - x3) をスロット単位で計算し、長さは x1（基底）に合わせ、
    さらにランダムで +1ペア / -1ペア / そのまま の長さ突然変異を加える。"""
    def strategy(candidate, population, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        S = population.shape[0]
        F = float(rng.uniform(*F_range))

        idxs = [i for i in range(S) if i != candidate]
        r1, r2, r3 = rng.choice(idxs, 3, replace=False)
        x1 = genome_to_pairs(population[r1], K_max)   # 基底（長さの基準）
        x2 = genome_to_pairs(population[r2], K_max)
        x3 = genome_to_pairs(population[r3], K_max)

        # x2 - x3: 短い方を amp=0 で長い方に合わせ、スロット単位で差分
        L = max(len(x2), len(x3))
        a2 = extend_pairs_zero(x2, L, tlen, rng)
        a3 = extend_pairs_zero(x3, L, tlen, rng)
        diff = [[a2[i][0] - a3[i][0], a2[i][1] - a3[i][1]] for i in range(L)]

        # 長さは x1 に合わせる → diff を len(x1) に整列して足し込む
        k1 = len(x1)
        diff_a = align_pairs_zero(diff, k1, tlen, rng)
        v = [[x1[i][0] + F * diff_a[i][0], x1[i][1] + F * diff_a[i][1]] for i in range(k1)]

        # 長さの突然変異: ランダムで -1ペア / そのまま / +1ペア
        choice = _rint(rng, 0, 3)
        if choice == 0 and len(v) > 1:
            v = shrink_pairs_by_amp(v, len(v) - 1)      # 小振幅順に1ペア削る
        elif choice == 2 and len(v) < K_max:
            v = add_random_pair(v, tlen, lim, rng)      # 実効振幅の新規ペアを追加

        return pairs_to_genome(v, K_max, tlen, lim, rng)
    return strategy


# ========= 目的関数（可変長・評価値 [-1,0]） =========
def apply_varlen(x_sample, genome, K_max):
    """先頭 k ペアを適用（位置重複は先勝ち）。戻り値: (適用後系列, 摂動数 k)。"""
    x_mod = np.copy(x_sample)
    tlen = x_mod.shape[0]
    k = int(round(genome[0]))
    k = max(1, min(k, K_max))
    used_positions = set()
    for j in range(k):
        pos = int(round(genome[1 + 2 * j]))
        pos = max(0, min(pos, tlen - 1))
        if pos in used_positions:
            continue
        used_positions.add(pos)
        x_mod[pos] = x_mod[pos] + genome[1 + 2 * j + 1]
    return x_mod, k


def objective_function_varlen(genome, model, x, label, lim, alpha, K_max):
    """評価値 = alpha*低下量 + (1-alpha)*摂動数（ともに [-1,0]、小さいほど良い）。
        低下量 = -(1 - probs[label])      （-1 = 完全に誤分類）
        摂動数 = -(1 - k / K_max)          （-1 = 摂動が最少）
    """
    P = np.asarray(genome, dtype=np.float64)
    single = (P.ndim == 1)
    P = P[:, None] if single else P     # (N, S)
    N, S = P.shape
    T = x.shape[0]

    org = x.reshape(1, T, 1).astype(np.float32)
    adv = np.repeat(org, S, axis=0)     # (S, T, 1)
    PT = P.T                            # (S, N)
    ks = np.zeros(S, dtype=np.float64)
    for s in range(S):
        k = int(round(PT[s, 0]))
        k = max(1, min(k, K_max))
        ks[s] = k
        used_positions = set()
        for j in range(k):
            pos = int(round(PT[s, 1 + 2 * j]))
            pos = max(0, min(pos, T - 1))
            if pos in used_positions:   # 重複は先勝ち
                continue
            used_positions.add(pos)
            adv[s, pos] = adv[s, pos] + PT[s, 1 + 2 * j + 1]

    probs = _predict_batch(model, adv)              # (S, nb_classes)
    misclass = -(1.0 - probs[:, label])             # [-1,0] 低下量
    npert = -(1.0 - ks / max(1, K_max))             # [-1,0] 摂動数
    vals = alpha * misclass + (1.0 - alpha) * npert # [-1,0]
    return float(vals[0]) if single else vals


class GenerationLogger:
    """可変長DEの世代ログ。best_value=1-conf（純粋な誤分類度）、num_perturbations=長さk。"""
    def __init__(self, model, x, K_max, label, history):
        self.model = model
        self.x = x
        self.K_max = K_max
        self.label = label
        self.history = history
        self.generation = 0

    def __call__(self, xk, convergence=None):
        self.generation += 1
        x_mod, k = apply_varlen(self.x, xk, self.K_max)
        probs = self.model(x_mod.reshape(1, -1, 1)).numpy()[0]
        self.history.append({
            "generation": self.generation,
            "best_value": 1.0 - float(probs[self.label]),
            "num_perturbations": k
        })


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


def make_initial_population_varlen(base_positions, K_max, tlen, lim):
    """初期個体行列 (INIT_POPULATION_SIZE, 1 + 2*K_max)。
    各個体の長さ k を [1, K_max] 一様で決め、活性先頭 k ペアに pos/amp、残りは amp=0。
    位置は前半 screening 上位を種に、後半ランダム。"""
    S = INIT_POPULATION_SIZE
    init_pop = np.zeros((S, 1 + 2 * K_max))

    for i in range(S):
        init_k = np.random.randint(1, K_max + 1)
        init_pop[i, 0] = init_k

        if i < S // 2 and base_positions:
            positions = list(base_positions[:K_max])
            while len(positions) < K_max:
                positions.append(np.random.randint(0, tlen))
        else:
            positions = [np.random.randint(0, tlen) for _ in range(K_max)]
        np.random.shuffle(positions)

        for j in range(K_max):
            init_pop[i, 1 + 2 * j] = positions[j]
            if j < init_k:
                init_pop[i, 1 + 2 * j + 1] = np.random.uniform(-lim * 0.3, lim * 0.3)
            else:
                init_pop[i, 1 + 2 * j + 1] = 0.0

    return init_pop


def process_single_sample(index, x_sample, y_sample, model,
                          K_max, lim, max_generations, strategy, tlen, alpha):
    """単一サンプルの処理（可変長DE・カスタム strategy・GPU一括推論）。
    変異は自前の可変長 rand1（長さは x1 基底＋±1ペア突然変異）、ループ/選択/収束は scipy。"""
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
    bounds = [(1, K_max)] + [(0, tlen - 1), (-lim, lim)] * K_max
    init_pop = make_initial_population_varlen(base_positions, K_max, tlen, lim)
    args = (model, x_sample, label, lim, alpha, K_max)
    logger = GenerationLogger(model, x_sample, K_max, label, generation_logs)
    strategy_fn = make_varlen_strategy(K_max, tlen, lim, F_range=(0.5, 1.0))

    result = differential_evolution(
        func=objective_function_varlen,
        bounds=bounds,
        args=args,
        strategy=strategy_fn,   # 可変長変異（カスタム callable）
        maxiter=max_generations,
        popsize=15,             # init を渡すので無視される
        disp=False,
        seed=42,
        callback=logger,
        init=init_pop,
        tol=0,
        atol=-1,
        polish=False,           # 離散長/位置の構造を崩さない
        vectorized=True         # 集団を1バッチ評価（GPU一括推論）
    )

    x_modified, optimized_num_perturbations = apply_varlen(x_sample, result.x, K_max)
    adv_x = x_modified.reshape(1, tlen, 1)
    probs = model(adv_x).numpy()[0]
    success = (int(np.argmax(probs)) != label)
    conf = float(probs[label])

    detailed_info = {
        "sample_id": index,
        "success": success,
        "message": getattr(result, "message", ""),
        "fun": result.fun,
        "nit": result.nit,
        "nfev": getattr(result, "nfev", 0),
        "confidence": conf,
        "optimized_num_perturbations": optimized_num_perturbations
    }

    for log_item in generation_logs:
        log_item["sample_id"] = index

    status = "成功" if success else "失敗"
    print(f"サンプル {index + 1} の処理完了 ({status}, 採用 k={optimized_num_perturbations}, conf={conf:.3f}, 世代={result.nit})")

    return {
        "index": index,
        "x_modified": x_modified,
        "detailed_info": detailed_info,
        "generation_logs": generation_logs,
        "nit": result.nit
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
                print(f"\n[{dataset}] 可変長DE(scipy/カスタム strategy): K_max={K_max}, alpha={alpha}, データ長: {tlen}")

                model = load_model(dataset, model_type)

                save_org_data(dataset, model_type, x_test)

                pred = model(x_test)
                test_acc = tf.metrics.SparseCategoricalAccuracy()
                test_acc(y_true, pred)
                org_acc = test_acc.result().numpy() * 100

                start = datetime.datetime.now()
                print(f"GPU一括推論版（vectorized）開始: サンプル数={len(x_test)}, 初期個体数={INIT_POPULATION_SIZE}")

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
    print(f"GPU一括推論版（vectorized・可変長DE/カスタム strategy）を使用: GPU数={len(gpus)}")
    print(f"初期個体数: {INIT_POPULATION_SIZE} (固定)")
    print(f"可変長DE: 個体は[長さk, (pos,amp)×K_max]。変異で長さは x1 基底＋ランダム±1ペア。評価=alpha*低下量+(1-alpha)*摂動数")

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

    end_all = datetime.datetime.now()
    print(f"開始日時：{start_all}")
    print(f"終了日時：{end_all}")