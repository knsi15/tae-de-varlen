import numpy as np
import random
import datetime
import os
import pandas as pd

#初期集団の生成
def generate_population(bounds, pop_size):
    popsize = pop_size * len(bounds)
    population = []

    for i in range(popsize):
        individual = []
        for j in range(len(bounds)):
            if(j % 2 == 0):
                lower, upper = bounds[j]
                value = random.randint(lower, upper)
            else:
                lower, upper = bounds[j]
                value = random.uniform(lower, upper)
            individual.append(value)
        population.append(individual)

    return population

#確認
# bounds = [(-5, 5), (-1, 1)] [位置、摂動]
# print(generate_population(bounds=bounds, pop_size=2))
#出力：2つ

#戦略
#rand1
def mutate_rand1(population, F, current_idx):
    if len(population) < 4:
        raise ValueError("rand/1の突然変異には、集団に少なくとも4個体が必要")
    indices = list(range(len(population)))
    indices.remove(current_idx) #自分自身は使わない、交叉で使う
    r1, r2, r3 = random.sample(indices, 3)  # 重複なしで選べる

    x1 = np.array(population[r1])
    x2 = np.array(population[r2])
    x3 = np.array(population[r3])

    mutant = x1 + F * (x2 - x3)
    return mutant.tolist()

#rand2
def mutate_rand2(population, F, current_idx):
    if len(population) < 6:
        raise ValueError("rand/2の突然変異には、集団に少なくとも6個体が必要")
    indices = list(range(len(population)))
    indices.remove(current_idx) #自分自身は使わない、交叉で使う
    r1, r2, r3, r4, r5 = random.sample(indices, 5)  # 重複なしで選べる

    x1 = np.array(population[r1])
    x2 = np.array(population[r2])
    x3 = np.array(population[r3])
    x4 = np.array(population[r4])
    x5 = np.array(population[r5])

    mutant = x1 + F * ((x2 - x3) + (x4 - x5))
    return mutant.tolist()

#best1
def mutate_best1(population, F, current_idx, fitness):
    if len(population) < 4:
        raise ValueError("best/1の突然変異には、集団に少なくとも4個体が必要")
    best_index = np.argmin(fitness)
    indices = list(range(len(population)))
    indices.remove(current_idx) #自分自身は使わない、交叉で使う
    indices.remove(best_index)
    r1, r2 = random.sample(indices, 2)  # 重複なしで選べる

    x_best = np.array(population[best_index])
    x1 = np.array(population[r1])
    x2 = np.array(population[r2])

    mutant = x_best + F * (x1 - x2)
    return mutant.tolist()

#best2
def mutate_best2(population, F, current_idx, fitness):
    if len(population) < 6:
        raise ValueError("best/2の突然変異には、集団に少なくとも6個体が必要")
    best_index = np.argmin(fitness)
    indices = list(range(len(population)))
    indices.remove(current_idx) #自分自身は使わない、交叉で使う
    indices.remove(best_index)
    r1, r2, r3, r4 = random.sample(indices, 4)  # 重複なしで選べる

    x_best = np.array(population[best_index])
    x1 = np.array(population[r1])
    x2 = np.array(population[r2])
    x3 = np.array(population[r3])
    x4 = np.array(population[r4])

    mutant = x_best + F * ((x1 - x2) + (x3 - x4))
    return mutant.tolist()

#currenttobest1
def mutate_currenttobest1(population, F, current_idx, fitness):
    if len(population) < 4:
        raise ValueError("currenttobest/1の突然変異には、集団に少なくとも4個体が必要")
    best_index = np.argmin(fitness)
    indices = list(range(len(population)))
    indices.remove(current_idx) #自分自身は使わない、交叉で使う
    indices.remove(best_index)
    r1, r2= random.sample(indices, 2)  # 重複なしで選べる

    x_best = np.array(population[best_index])
    x_current = np.array(population[current_idx])
    x1 = np.array(population[r1])
    x2 = np.array(population[r2])

    mutant = x_current + F * ((x_best - x_current) + (x1 - x2))
    return mutant.tolist()

#randtobest1
def mutate_randtobest1(population, F, current_idx, fitness):
    if len(population) < 5:
        raise ValueError("randtobest/1の突然変異には、集団に少なくとも5個体が必要")
    best_index = np.argmin(fitness)
    indices = list(range(len(population)))
    indices.remove(current_idx) #自分自身は使わない、交叉で使う
    indices.remove(best_index)
    r1, r2, r3 = random.sample(indices, 3)  # 重複なしで選べる

    x_best = np.array(population[best_index])
    x1 = np.array(population[r1])
    x2 = np.array(population[r2])
    x3 = np.array(population[r3])

    mutant = x1 + F * ((x_best - x1) + (x2 - x3))
    return mutant.tolist()

#currenttobest2
def mutate_currenttobest2(population, F, current_idx, fitness):
    if len(population) < 6:
        raise ValueError("currenttobest/2の突然変異には、集団に少なくとも6個体が必要で")
    best_index = np.argmin(fitness)
    indices = list(range(len(population)))
    indices.remove(current_idx) #自分自身は使わない、交叉で使う
    indices.remove(best_index)
    r1, r2, r3, r4 = random.sample(indices, 4)  # 重複なしで選べる

    x_best = np.array(population[best_index])
    x_current = np.array(population[current_idx])
    x1 = np.array(population[r1])
    x2 = np.array(population[r2])
    x3 = np.array(population[r3])
    x4 = np.array(population[r4])

    mutant = x_current + F * ((x_best - x_current) + (x1 - x2) + (x3 - x4))
    return mutant.tolist()

#交叉
#binomial
def crossover_binomial(target, mutant, CR):
    D = len(target)
    trial = []
    rand_index = random.randint(0, D - 1)

    for i in range(D):
        if random.random() < CR or i == rand_index: #交叉率以下なら変異ベクトル, 1つは変異ベクトルから
            trial.append(mutant[i])
        else: #以上なら親個体から
            trial.append(target[i])
    
    return trial

#exponential
def crossover_exponential(target, mutant, CR):
    D = len(target)
    trial = target.copy() 
    trial = list(trial)
    start = random.randint(0, D - 1)
    L = 1

    while random.random() < CR and L < D: # 交叉ブロックの長さ、満たすまでループ
        L += 1

    for i in range(L): # mutantの値を部分的にコピー
        index = (start + i) % D
        trial[index] = mutant[index]

    return trial

mutation_strategies = {
    "rand1": mutate_rand1,
    "rand2": mutate_rand2,
    "best1": mutate_best1,
    "best2": mutate_best2,
    "currenttobest1": mutate_currenttobest1,
    "currenttobest2": mutate_currenttobest2,
    "randtobest1": mutate_randtobest1,
}

crossover_strategies = {
    "binomial": crossover_binomial,
    "exponential": crossover_exponential
}

#選択
def select(target, trial, fitness_func):
    target_fitness = fitness_func(target)
    trial_fitness = fitness_func(trial)
    if trial_fitness < target_fitness:  # 最小化問題の場合
        return trial, trial_fitness
    else:
        return target, target_fitness

# 目的関数の定義
def objective_function_multiitem(perturbations, model, x, y, nb_classes):

    #label = np.where(y > 0) # xのクラスラベルの取得
    label = np.argmax(y)
    x = np.copy(x)
    x = x.reshape(1, x.shape[0],x.shape[1])

    # 遺伝子型を用いて、敵対的サンプルを生成
    plist = [perturbations[i:i+2] for i in range(0, len(perturbations), 2)]
    for p in plist:
        pos = int(p[0])
        val = p[1]
        x[0, pos] = x[0, pos] + val

    # モデルを通して分類確率を取得
    prep = model(x)

    return -(1-prep.numpy()[0, label]) #誤分類確率を負に変換、値が小さいほど良い個体

#main
def differential_evolution(
    fitness_func, #目的関数
    bounds,
    pop_size, #popsize = pop_size * len(bounds)
    generations, #最大世代数
    F, #スケーリングファクター
    CR, #交叉率
    strategy,  # 例: "rand1bin", "best2exp"
    seed=None, #乱数値
    model=None,
    x=None,
    y=None, 
    nb_classes=None, #クラス
    tol=0.05, #早期終了条件: 閾値
    patience=20 #早期終了条件: 停滞数
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    mutation_name, crossover_name = strategy[:-3], strategy[-3:] #末尾3つで切り分け
    if crossover_name == "bin":
        crossover_name = "binomial"
    elif crossover_name == "exp":
        crossover_name = "exponential"
    else:
        raise ValueError("binかexpで指定")        

    population = generate_population(bounds, pop_size)

    fitness = []
    for ind in population: #適応度を測る
        score =fitness_func(ind, model, x, y, nb_classes)
        fitness.append(score)

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    no_improve_count = 0

    for gen in range(generations): #世代数
        new_population = []
        new_fitness = []
        
        F = random.uniform(F[0], F[1])

        for i in range(pop_size): #個体1つずつ進化
            # 突然変異
            if "best" in mutation_name:
                mutant = mutation_strategies[mutation_name](population, F, i, fitness)
            else:
                mutant = mutation_strategies[mutation_name](population, F, i)

            # 交叉
            trial = crossover_strategies[crossover_name](population[i], mutant, CR)

            # 選択
            selected, selected_fitness = select( #個体1つずつ適応度を測る
                population[i],
                trial,
                lambda z: fitness_func(z, model, x, y, nb_classes)
            )

            new_population.append(selected)
            new_fitness.append(selected_fitness)

        population = new_population
        fitness = new_fitness

        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]

        best_idx = np.argmin(fitness)
        print(f"Generation {gen+1}: Best fitness = {fitness[best_idx]}")

        # 収束判定
        if abs(current_best_fitness - best_fitness) < tol:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Converged after {gen+1} generations.") #収束
                break
        else:
            no_improve_count = 0
            best_fitness = current_best_fitness

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx] #良い個体と、適応度を返す

#excel保存
def save_result_to_excel(
        filename: str,
        dataset_name: str,
        model: str,
        strategy: str,
        eval_func_name: str,
        original_accuracy: float,
        attack_accuracy: float,
        start_time: datetime,
        end_time: datetime,
        mae: float,
        mse: float,
        rmse: float
):
    duration = (end_time -start_time).total_seconds()

    result_row = {
        "開始時間": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "終了時間": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "実行時間（秒）": duration,
        "モデル名": model,
        "戦略": strategy,
        "データセット名": dataset_name,
        "評価関数": eval_func_name,
        "元精度": original_accuracy,
        "攻撃後精度": attack_accuracy,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

    file_exists = os.path.exists(filename)

    if file_exists:
        df = pd.read_excel(filename)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])

    df.to_excel(filename, index=False)