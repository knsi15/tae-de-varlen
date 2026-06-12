import os
import sys
import random
import numpy as np
import tensorflow as tf
from deap import base, creator, tools
from keras.utils import to_categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.preprocess import load_data, load_model, save_org_data, save_ae_data
from lib.differential_evolution import (
    mutate_rand1,
    crossover_exponential,
    objective_function_multiitem,
)

# モジュールレベルで1回だけ定義
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

# 初期個体定義
def init_individual(bounds):
    ind = []
    for j, (lower, upper) in enumerate(bounds):
        if j % 2 == 0:
            ind.append(float(random.randint(int(lower), int(upper))))
        else:
            ind.append(random.uniform(lower, upper))
    return creator.Individual(ind)

# DEの型
def run_de_on_sample(index, x_sample, y_sample, model, nb_classes, bounds,
                     population_size, generations, F_range, CR, tol, patience):
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, bounds)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=population_size)

    # 初期集団の評価
    for ind in pop:
        fit = objective_function_multiitem(list(ind), model, x_sample, y_sample, nb_classes)
        ind.fitness.values = (fit,)

    best_fitness = min(ind.fitness.values[0] for ind in pop)
    no_improve_count = 0

    for gen in range(generations):
        F = random.uniform(*F_range)
        raw_pop = [list(ind) for ind in pop]  # 世代開始時の集団スナップショット

        new_pop = []
        for i, target in enumerate(pop):
            # 突然変異: rand/1
            mutant = mutate_rand1(raw_pop, F, i)

            # 交叉: exponential
            trial = crossover_exponential(list(target), mutant, CR)

            # 選択: greedy（trialが良ければ入れ替え）
            trial_fit = objective_function_multiitem(trial, model, x_sample, y_sample, nb_classes)
            if trial_fit < target.fitness.values[0]:
                new_ind = creator.Individual(trial)
                new_ind.fitness.values = (trial_fit,)
                new_pop.append(new_ind)
            else:
                new_pop.append(target)

        pop = new_pop
        current_best_fitness = min(ind.fitness.values[0] for ind in pop)
        print(f"  [sample {index}] Gen {gen + 1}: best = {current_best_fitness:.6f}")

        if abs(current_best_fitness - best_fitness) < tol:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  Converged at generation {gen + 1}.")
                break
        else:
            no_improve_count = 0
            best_fitness = current_best_fitness

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return list(best_ind)

# main
def main(dataset, model_type):
    x_test, y_test = load_data(dataset, is_test=True)

    nb_classes = len(np.unique(y_test))
    y_true = y_test
    y_test = to_categorical(y_test)
    tlen = len(x_test[0])

    lim = 0.3
    n_perturbations = 5
    bounds = [(0, tlen - 1), (-lim, lim)] * n_perturbations

    model = load_model(dataset, model_type)
    save_org_data(dataset, model_type, x_test)

    # 攻撃前の精度
    pred = model(x_test)
    test_acc = tf.metrics.SparseCategoricalAccuracy()
    test_acc(y_true, pred)
    org_acc = test_acc.result().numpy() * 100

    # DE パラメータ（mdeattack_timeseries.py に合わせる）
    population_size = 300
    generations = 200
    F_range = (0.5, 1.0)
    CR = 0.7
    tol = 0.05
    patience = 20
    seed = 42

    random.seed(seed)
    np.random.seed(seed)

    for index in range(len(x_test)):
        best_perturbations = run_de_on_sample(
            index=index,
            x_sample=x_test[index],
            y_sample=y_test[index],
            model=model,
            nb_classes=nb_classes,
            bounds=bounds,
            population_size=population_size,
            generations=generations,
            F_range=F_range,
            CR=CR,
            tol=tol,
            patience=patience,
        )

        # 最良個体の摂動をテストデータに適用
        bplist = [best_perturbations[i:i + 2] for i in range(0, len(best_perturbations), 2)]
        for p in bplist:
            pos = int(p[0])
            val = p[1]
            x_test[index][pos][0][0] = x_test[index][pos][0][0] + val

    save_ae_data(dataset, model_type, x_test)

    # 攻撃後の精度
    pred = model(x_test)
    test_acc = tf.metrics.SparseCategoricalAccuracy()
    test_acc(y_true, pred)
    ae_acc = test_acc.result().numpy() * 100
    print(f"{dataset},{org_acc},{ae_acc}")

if __name__ == "__main__":
    main("Coffee", "fcn")
