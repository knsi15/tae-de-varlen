import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf

random.seed(10)

# データ・モデルのベースディレクトリ。
# lib/preprocess.py を基準にリポジトリの 1 つ上 (tae/) 直下の data / models を指す。
# __file__ から導出するので Windows / WSL のどちらでも同じ実体を解決できる。
# 環境変数 TAE_DATA_DIR / TAE_MODELS_DIR で明示的に上書き可能。
_TAE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("TAE_DATA_DIR", os.path.join(_TAE_ROOT, "data"))
MODELS_DIR = os.environ.get("TAE_MODELS_DIR", os.path.join(_TAE_ROOT, "models"))

# データ前処理  ################################################
def preprocess(data_df, norm=False):
  # ラベルとデータを分割
  y_data = data_df.iloc[:, 0].values
  x_data = data_df.iloc[:, 1:].values

  # 標準化
  if norm:
    for i in range(x_data.shape[0]):
      x_mean = x_data[i].mean()
      x_std = x_data[i].std()
      x_data[i] = (x_data[i] - x_mean) / x_std

  # 入力できる形式に変換
  x_data = x_data.reshape(x_data.shape + (1,1,))
  x_data = x_data.astype(np.float32)
  nb_classes = len(np.unique(y_data)) #クラス数
  if nb_classes > 1:
    y_data = (y_data - y_data.min())/(y_data.max()-y_data.min())*(nb_classes-1)
  else:
    y_data = y_data - y_data.min()

  return x_data, y_data


# データ保存  ################################################
def save_org_data(dataset, modal_type,data):
  data = data.reshape(data.shape[0],data.shape[1])
  np.savetxt(os.path.join(DATA_DIR, dataset, dataset + '_' + modal_type + '_TEST_ORG.tsv'), data, delimiter='\t')

# データ保存  ################################################
def save_ae_data(dataset, modal_type,data):
  data = data.reshape(data.shape[0],data.shape[1])
  np.savetxt(os.path.join(DATA_DIR, dataset, dataset + '_' + modal_type + '_TEST_AE.tsv'), data, delimiter='\t')

# データ読み込み  ################################################
def load_data(dataset, is_test=False, norm=False):
  if is_test:
    test_df = pd.read_table(os.path.join(DATA_DIR, dataset, dataset + '_TEST.tsv'), header=None)
    x_test, y_test = preprocess(test_df, norm)
    return x_test, y_test

  else:
    train_df = pd.read_table(os.path.join(DATA_DIR, dataset, dataset + '_TRAIN.tsv'), header=None)
    x_train, y_train = preprocess(train_df, norm)
    return x_train, y_train


# バッチ生成 ################################################
def batch(x, y, size, shuffle=False):

  index_list = list(range(len(y)))
  if shuffle:
    random.shuffle(index_list)

  batched_x = []
  batched_y = []
  for idx in range(0, len(index_list), size):
    batch_indices = np.array(index_list[idx: idx + size])
    batched_x.append(x[batch_indices])
    batched_y.append(y[batch_indices])

  return batched_x, batched_y


# モデル読み込み  ################################################
def load_model(dataset, model_type):
  model_dir = os.path.join(MODELS_DIR, model_type, 'UCRArchive_2018_itr_8')
  model = tf.keras.models.load_model(os.path.join(model_dir, dataset, 'best_model.hdf5'))
  return model