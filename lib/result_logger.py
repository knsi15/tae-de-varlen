"""
result_saver.py - 結果保存モジュール
Excel保存、詳細ログ、世代ログ、プロット生成などの結果保存処理を提供する。
"""

import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'


def create_output_dirs(date_str, output_base_dir="output"):
    """
    日付ごとの出力フォルダを作成する
    output/YYYYMMDD/
    output/YYYYMMDD/plot
    output/YYYYMMDD/generation_plot
    """
    date_dir = os.path.join(output_base_dir, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)

    plot_dir = os.path.join(date_dir, "plot")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    gen_plot_dir = os.path.join(date_dir, "generation_plot")
    if not os.path.exists(gen_plot_dir):
        os.makedirs(gen_plot_dir)

    return date_dir, plot_dir, gen_plot_dir


def get_unique_dir_name(base_dir, date_str):
    """
    指定された日付文字列に対応するディレクトリが既に存在する場合、
    _1, _2, ... のようにサフィックスを付けて一意な名前を生成する。
    """
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


def backup_code(target_dir, source_file=None):
    """
    実行コード(自分自身)を保存先にコピーする

    Args:
        target_dir: コピー先ディレクトリ
        source_file: コピー元ファイルパス (デフォルト: 呼び出し元の__file__)
    """
    if source_file is None:
        import inspect
        frame = inspect.stack()[1]
        source_file = os.path.abspath(frame.filename)
    else:
        source_file = os.path.abspath(source_file)

    filename = os.path.basename(source_file)
    dst_path = os.path.join(target_dir, filename)
    try:
        shutil.copy2(source_file, dst_path)
        print(f"コード記述のバックアップを作成しました: {dst_path}")
    except Exception as e:
        print(f"バックアップ作成エラー: {e}")


def save_result_to_excel(
    filename: str,
    dataset_name: str,
    model: str,
    eval_func_name: str,
    strategy: str,
    original_accuracy: float,
    attack_accuracy: float,
    start_time,
    end_time,
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


def save_detailed_results(detailed_results, output_dir, dataset, model_type, strategy, func, lim):
    """
    サンプルごとの詳細な実行結果をログ保存する
    """
    if not detailed_results:
        return

    filename = f"detailed_log_{dataset}_{model_type}_{strategy}_{func}_lim{lim}.xlsx"
    filepath = os.path.join(output_dir, filename)

    df = pd.DataFrame(detailed_results)
    df.to_excel(filepath, index=False)
    print(f"詳細ログ保存完了: {filepath}")


def save_generation_log(generation_logs, output_dir, dataset, model_type, strategy, func, lim):
    """
    世代ごとの評価値を保存する
    """
    if not generation_logs:
        return

    filename = f"generation_log_{dataset}_{model_type}_{strategy}_{func}_lim{lim}.csv"
    filepath = os.path.join(output_dir, filename)

    df = pd.DataFrame(generation_logs)
    df.to_csv(filepath, index=False)
    print(f"世代ログ保存完了: {filepath}")


def plot_generation_history(generation_logs, output_dir, dataset, model_type, strategy, func, lim, maxiter=50):
    """
    世代ごとの信頼度推移をプロットする
    縦軸: 信頼度(0-1), 横軸: 世代(0-maxiter)
    """
    if not generation_logs:
        return

    # サンプルごとにデータを整理
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

        plt.title(f'Sample {sample_id}\n{dataset} {model_type} {strategy} {func}')
        plt.xlabel('Generation')
        plt.ylabel('Confidence')
        plt.ylim(0, 1.05)
        plt.xlim(0, maxiter)
        plt.grid(True, linestyle='--', alpha=0.7)

        filename = f"generation_history_{dataset}_{model_type}_{strategy}_{func}_lim{lim}_{sample_id}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"世代推移プロット保存完了: {output_dir}")


def align_to_length(series_array: np.ndarray, target_length: int) -> np.ndarray:
    array_1d = np.asarray(series_array, dtype=np.float32).reshape(-1)
    current_length = array_1d.shape[0]
    if target_length is None or current_length == target_length:
        return array_1d
    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(array_1d, (0, pad_width), mode='edge')
    return array_1d[:target_length]


def create_comparison_plot(dataset, model_type, strategy, func, index, org_data, ae_data, model, current_date, lim, true_label=None, output_dir=None):
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

            org_pred = model.predict(org_processed, verbose=0)
            ae_pred = model.predict(ae_processed, verbose=0)

            org_class = np.argmax(org_pred[0])
            ae_class = np.argmax(ae_pred[0])
            if org_class == ae_class:
                classification_result = "same"
                result_color = "green"
            else:
                classification_result = "different"
                result_color = "red"

            # 真のラベルとの比較
            if true_label is not None:
                true_label_int = int(true_label)
                ae_correct = (ae_class == true_label_int)
                true_label_result = f"True: {true_label_int}, AE pred: {ae_class} ({'correct' if ae_correct else 'misclassified'})"
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
        plt.title(f'{dataset} - {model_type.upper()} Model - {strategy} - {func} - Sample {index+1}', fontsize=14, fontweight='bold')
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
        filename = f'{current_date}_{dataset}_{model_type}_{strategy}_{func}_lim{lim}_{index+1}.png'
        filepath = os.path.join(dataset_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"画像保存完了: {filepath}")
    except Exception as e:
        print(f"画像生成エラー (Sample {index+1}): {str(e)}")
        plt.close()
