"""
プロット関連のユーティリティモジュール。
- 元波形と摂動波形の比較プロット
- 世代ごとの信頼度推移プロット
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'


def align_to_length(series_array: np.ndarray, target_length: int) -> np.ndarray:
    """
    1次元時系列を target_length に合わせる。
    短ければ末尾エッジでパディング、長ければ切り詰める。
    """
    array_1d = np.asarray(series_array, dtype=np.float32).reshape(-1)
    current_length = array_1d.shape[0]
    if target_length is None or current_length == target_length:
        return array_1d
    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(array_1d, (0, pad_width), mode='edge')
    return array_1d[:target_length]


def create_comparison_plot(dataset, model_type, strategy, func, index, org_data, ae_data,
                           model, current_date, lim, true_label=None, output_dir=None):
    """
    元データと摂動後データを重ねてプロットし、分類結果を併記する。
    """
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

            if true_label is not None:
                true_label_int = int(true_label)
                ae_correct = (ae_class == true_label_int)
                true_label_result = (
                    f"True: {true_label_int}, AE pred: {ae_class} "
                    f"({'correct' if ae_correct else 'misclassified'})"
                )
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
        plt.title(
            f'{dataset} - {model_type.upper()} Model - {strategy} - {func} - Sample {index+1}',
            fontsize=14, fontweight='bold'
        )
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
            base_plot_dir = 'plot'
        os.makedirs(base_plot_dir, exist_ok=True)

        dataset_dir = os.path.join(base_plot_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        filename = f'{current_date}_{dataset}_{model_type}_{strategy}_{func}_lim{lim}_{index+1}.png'
        filepath = os.path.join(dataset_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"画像保存完了: {filepath}")
    except Exception as e:
        print(f"画像生成エラー (Sample {index+1}): {str(e)}")
        plt.close()


def plot_generation_history(generation_logs, output_dir, dataset, model_type, strategy, func,
                            lim, maxiter=50):
    """
    世代ごとの信頼度推移をサンプル単位でプロットする。
    縦軸: 信頼度(0-1), 横軸: 世代(0-maxiter)
    """
    if not generation_logs:
        return

    sample_logs = {}
    for log in generation_logs:
        sample_logs.setdefault(log["sample_id"], []).append(log)

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
