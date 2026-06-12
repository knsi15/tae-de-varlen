"""
出力ディレクトリの作成とコードのバックアップを担当するモジュール。
"""

import os
import shutil

OUTPUT_BASE_DIR = "output"


def create_output_dirs(date_str, base_dir=OUTPUT_BASE_DIR):
    """
    日付ごとの出力フォルダを作成する。
    output/YYYYMMDD/
    output/YYYYMMDD/plot
    output/YYYYMMDD/generation_plot
    """
    date_dir = os.path.join(base_dir, date_str)
    plot_dir = os.path.join(date_dir, "plot")
    gen_plot_dir = os.path.join(date_dir, "generation_plot")

    for d in (date_dir, plot_dir, gen_plot_dir):
        os.makedirs(d, exist_ok=True)

    return date_dir, plot_dir, gen_plot_dir


def get_unique_dir_name(base_dir, date_str):
    """
    指定された日付文字列に対応するディレクトリが既に存在する場合、
    _1, _2, ... のようにサフィックスを付けて一意な名前を生成する。
    """
    if not os.path.exists(os.path.join(base_dir, date_str)):
        return date_str

    counter = 1
    while True:
        new_date_str = f"{date_str}_{counter}"
        if not os.path.exists(os.path.join(base_dir, new_date_str)):
            return new_date_str
        counter += 1


def backup_code(target_dir, source_file):
    """
    実行コード(source_file)を保存先にコピーする。

    Args:
        target_dir: コピー先のディレクトリ
        source_file: バックアップ対象ファイル(通常は呼び出し側の __file__)
    """
    src_path = os.path.abspath(source_file)
    dst_path = os.path.join(target_dir, os.path.basename(src_path))
    try:
        shutil.copy2(src_path, dst_path)
        print(f"コード記述のバックアップを作成しました: {dst_path}")
    except Exception as e:
        print(f"バックアップ作成エラー: {e}")
