import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import pandas as pd
import os
import gc
import multiprocessing
from multiprocessing import Pool, cpu_count

from deap import base, creator, tools

import sys
sys.path.append("../lib")

from lib.preprocess import load_data, load_model, save_org_data, save_ae_data
from lib.result_logger import (
    create_output_dirs,
    get_unique_dir_name,
    backup_code,
    save_result_to_excel,
    save_detailed_results,
    save_generation_log,
    plot_generation_history,
    create_comparison_plot,
)
from keras.utils import to_categorical

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# DEAP setup
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

# グローバル変数
MAX_PERTURBATIONS = 15
GA_POP_SIZE = 300
MAX_GENERATIONS = 200
LIM = 0.3
GLOBAL_START_DATE = None
OUTPUT_BASE_DIR = "output"

# GPU認識
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


# 個体生成
def create_individual(tlen, lim):
    n = random.randint(1, 5)
    ind = []
    for _ in range(n):
        ind.append(random.randint(0, tlen - 1))
        ind.append(random.uniform(-lim, lim))
    return creator.Individual(ind)

# 評価関数
def evaluate_individual(individual, model, x, y, lim):
    label = np.argmax(y)
    n_pairs = len(individual) // 2

    x_copy = np.copy(x)
    adv_x = x_copy.reshape(1, x_copy.shape[0], x_copy.shape[1])

    for i in range(n_pairs):
        pos = int(individual[i * 2]) % adv_x.shape[1]
        amp = float(individual[i * 2 + 1])
        adv_x[0, pos] += amp

    probs = model(adv_x).numpy()

    alpha = 0.6
    misclass_score = -(1 - probs[0, label])
    npert_score = -1.0 / max(n_pairs, 1)
    return (alpha * misclass_score + (1 - alpha) * npert_score,)

# 交叉
def crossover_exp(ind1, ind2, CR=0.7):
    """
    DEのexponential crossoverをペア単位で適用。
    j_rand から始まり、random() < CR の間だけ連続してペアを交換する。
    個体の長さは変わらない（長さ変化は mutate_vl が担当）。
    """
    pairs1 = [[ind1[i], ind1[i + 1]] for i in range(0, len(ind1), 2)]
    pairs2 = [[ind2[i], ind2[i + 1]] for i in range(0, len(ind2), 2)]

    if not pairs1 or not pairs2:
        return ind1, ind2

    n1, n2 = len(pairs1), len(pairs2)
    new_pairs1 = [p[:] for p in pairs1]
    new_pairs2 = [p[:] for p in pairs2]

    # ind1 の j_rand から連続して ind2 のペアを採用
    j_rand = random.randint(0, n1 - 1)
    j = j_rand
    new_pairs1[j] = pairs2[j % n2][:]          # j_rand は必ず交換
    j = (j + 1) % n1
    while j != j_rand and random.random() < CR:
        new_pairs1[j] = pairs2[j % n2][:]
        j = (j + 1) % n1

    # ind2 の j_rand から連続して ind1 のペアを採用
    j_rand2 = random.randint(0, n2 - 1)
    j = j_rand2
    new_pairs2[j] = pairs1[j % n1][:]
    j = (j + 1) % n2
    while j != j_rand2 and random.random() < CR:
        new_pairs2[j] = pairs1[j % n1][:]
        j = (j + 1) % n2

    ind1[:] = [v for p in new_pairs1 for v in p]
    ind2[:] = [v for p in new_pairs2 for v in p]
    return ind1, ind2

# 変異
def mutate_vl(individual, tlen, lim, p_add=0.3, p_remove=0.2, max_perturbs=30):
    n_pairs = len(individual) // 2
    r = random.random()
    sigma_amp = lim * 0.1

    if r < p_add and n_pairs < max_perturbs:
        individual.append(random.randint(0, tlen - 1))
        individual.append(random.uniform(-lim, lim))
    elif r < p_add + p_remove and n_pairs > 1:
        idx = random.randint(0, n_pairs - 1)
        del individual[idx * 2:idx * 2 + 2]
    else:
        if n_pairs > 0:
            idx = random.randint(0, n_pairs - 1)
            if random.random() < 0.5:
                individual[idx * 2] = random.randint(0, tlen - 1)
            else:
                individual[idx * 2 + 1] = float(
                    max(-lim, min(lim, individual[idx * 2 + 1] + random.gauss(0, sigma_amp)))
                )
    return (individual,)

def run_deap_ga(model, x_sample, y_sample, tlen, lim, max_generations, pop_size):
    # deapで処理する操作の登録（評価関数とか戦略）
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, tlen=tlen, lim=lim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual,
                     model=model, x=x_sample, y=y_sample, lim=lim)
    toolbox.register("mate", crossover_exp)
    toolbox.register("mutate", mutate_vl, tlen=tlen, lim=lim)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    generation_logs = []

    # 世代ごとの処理
    for gen in range(max_generations):
        elite = toolbox.clone(tools.selBest(pop, 1)[0])

        offspring = toolbox.select(pop, pop_size - 1)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring + [elite]

        best = tools.selBest(pop, 1)[0]
        generation_logs.append({
            "generation": gen + 1,
            "best_value": -best.fitness.values[0],
            "num_perturbations": len(best) // 2
        })

    best = tools.selBest(pop, 1)[0]
    return best, generation_logs, max_generations

# 単一サンプル処理
def process_single_sample(index, x_sample, y_sample, dataset, model_type,
                          nb_classes, lim, max_generations, tlen, pop_size):
    configure_gpu()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    random.seed(42 + index)
    np.random.seed(42 + index)

    print(f"サンプル {index + 1} の処理を開始...")

    model = load_model(dataset, model_type)

    best_individual, generation_logs, nit = run_deap_ga(
        model=model,
        x_sample=x_sample,
        y_sample=y_sample,
        tlen=tlen,
        lim=lim,
        max_generations=max_generations,
        pop_size=pop_size
    )

    n_pairs = len(best_individual) // 2

    detailed_info = {
        "sample_id": index,
        "fun": best_individual.fitness.values[0],
        "nit": nit,
        "nfev": nit * pop_size,
        "optimized_num_perturbations": n_pairs
    }

    for log_item in generation_logs:
        log_item["sample_id"] = index

    x_modified = np.copy(x_sample)
    used_positions = set()
    for i in range(n_pairs):
        pos = int(best_individual[i * 2]) % tlen
        amp = float(best_individual[i * 2 + 1])
        if pos in used_positions:
            continue
        used_positions.add(pos)
        x_modified[pos][0][0] = x_modified[pos][0][0] + amp

    del model
    gc.collect()

    print(f"サンプル {index + 1} の処理完了 (摂動数: {n_pairs})")

    return {
        "index": index,
        "x_modified": x_modified,
        "detailed_info": detailed_info,
        "generation_logs": generation_logs,
        "nit": nit
    }

# メイン
def main(dataset, model_type, lim=0.1, num_workers=None):
    current_date_dir, current_plot_dir, current_gen_plot_dir = create_output_dirs(GLOBAL_START_DATE)

    gen = []
    detailed_results = []
    all_generation_logs = []

    x_test, y_test = load_data(dataset, is_test=True)

    nb_classes = len(np.unique(y_test))
    y_true = y_test
    y_test = to_categorical(y_test)
    tlen = len(x_test[0])

    print(f"\n[{dataset}] 最大世代数: {MAX_GENERATIONS}, 個体数: {GA_POP_SIZE}, データ長: {tlen}")

    model = load_model(dataset, model_type)
    save_org_data(dataset, model_type, x_test)

    pred = model(x_test)
    test_acc = tf.metrics.SparseCategoricalAccuracy()
    test_acc(y_true, pred)
    org_acc = test_acc.result().numpy() * 100

    start = datetime.datetime.now()

    process_args = [
        (index, x_test[index], y_test[index], dataset, model_type,
         nb_classes, lim, MAX_GENERATIONS, tlen, GA_POP_SIZE)
        for index in range(len(x_test))
    ]

    gpus = tf.config.list_physical_devices('GPU')
    has_gpu = len(gpus) > 0

    if num_workers is None:
        num_workers_actual = min(8 if has_gpu else cpu_count(), len(x_test))
    else:
        num_workers_actual = min(num_workers, len(x_test))

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

    avg_gen = np.mean(gen) if gen else 0
    avg_np = np.mean(optimized_np_list) if optimized_np_list else 0

    save_result_to_excel(
        filename="results.xlsx",
        dataset_name=dataset,
        model=model_type,
        eval_func_name="misclass_with_npert_vl",
        strategy="ga",
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
        max_perturbations=MAX_PERTURBATIONS,
        output_dir=current_date_dir
    )

    save_detailed_results(detailed_results, current_date_dir, dataset, model_type,
                          "ga", "misclass_with_npert_vl", lim)
    save_generation_log(all_generation_logs, current_date_dir, dataset, model_type,
                        "ga", "misclass_with_npert_vl", lim)
    plot_generation_history(all_generation_logs, current_gen_plot_dir, dataset, model_type,
                            "ga", "misclass_with_npert_vl", lim, maxiter=MAX_GENERATIONS)

    print(f"画像生成開始: {dataset} - {model_type}")
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
                strategy="ga",
                func="misclass_with_npert_vl",
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

        print(f"画像生成完了: {dataset} - {model_type} ({num_rows}枚)")

    except Exception as e:
        print(f"画像生成エラー ({dataset}_{model_type}): {str(e)}")

    del model
    gc.collect()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    configure_gpu()

    base_date_str = datetime.datetime.now().strftime("%Y%m%d")
    GLOBAL_START_DATE = get_unique_dir_name(OUTPUT_BASE_DIR, base_date_str)

    d_dir, _, _ = create_output_dirs(GLOBAL_START_DATE)
    backup_code(d_dir, __file__)

    gpus = tf.config.list_physical_devices('GPU')

    start_all = datetime.datetime.now()

    main("BeetleFly", "fcn", lim=LIM)
    main("Car", "fcn", lim=LIM)
    main("Coffee", "fcn", lim=LIM)
    main("Computers", "fcn", lim=LIM)
    main("ECG200", "fcn", lim=LIM)
    main("ToeSegmentation2", "fcn", lim=LIM)
    main("ShapeletSim", "fcn", lim=LIM)

    end_all = datetime.datetime.now()
    print(f"開始日時：{start_all}")
    print(f"終了日時：{end_all}")
