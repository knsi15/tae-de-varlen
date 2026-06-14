"""
可変長DE（自前ループ版）。
個体は可変長リスト [[pos, amp], ...]（摂動数 k が個体ごとに異なる）。
固定幅エンコーディングを使わず、変異・交叉で素直に長さを扱う。

評価はモデル非依存にするため、呼び出し側が「集団全体を1バッチで評価する」
fitness_batch(population) -> np.ndarray を渡す（GPU一括推論を保つための設計）。

長さ合わせの方針（「もともと無かった所に乱数位置を置かない」）:
- 差分 x2-x3 で足りないスロットは「相手の位置・振幅0」で補い、位置差を0にする
- 差分ベクトルを x1 長へ伸ばすときは [0,0]（no-op）で補う
- 交叉でターゲットに無いスロットは mutant から取る（phantom を作らない）
"""

import numpy as np
import random


def shrink_pairs_by_amp(pairs, n):
    """長い→短い: |amp| の小さい順に並べ替え、先頭 n 個を残す（小振幅を残す＝非知覚性重視）。"""
    order = sorted(range(len(pairs)), key=lambda i: abs(pairs[i][1]))
    return [list(pairs[i]) for i in order[:max(1, n)]]


def align_diff(diff, n):
    """差分ベクトルを長さ n に整列。短ければ [0,0]（位置も振幅も動かさない no-op）で伸長、
    長ければ |amp| 小さい順に短縮。"""
    if len(diff) < n:
        out = [list(d) for d in diff]
        while len(out) < n:
            out.append([0.0, 0.0])
        return out
    if len(diff) > n:
        return shrink_pairs_by_amp(diff, n)
    return [list(d) for d in diff]


def clip_pairs(pairs, tlen, lim):
    """pos を [0, tlen-1]、amp を [-lim, lim] にクリップ。"""
    out = []
    for pos, amp in pairs:
        out.append([min(max(int(round(pos)), 0), tlen - 1), min(max(amp, -lim), lim)])
    return out


# ========= 初期集団 =========
def generate_population_varlen(pop_size, K_max, tlen, lim, base_positions=None):
    """各個体の長さ k を [1, K_max] 一様で決め、リスト [[pos, amp], ...] を作る。
    位置は前半 screening 上位を種に、後半ランダム。"""
    base_positions = list(base_positions) if base_positions else []
    population = []
    for i in range(pop_size):
        k = random.randint(1, K_max)
        if i < pop_size // 2 and base_positions:
            positions = list(base_positions[:K_max])
            while len(positions) < K_max:
                positions.append(random.randint(0, tlen - 1))
        else:
            positions = [random.randint(0, tlen - 1) for _ in range(K_max)]
        random.shuffle(positions)
        ind = [[positions[j], random.uniform(-lim * 0.3, lim * 0.3)] for j in range(k)]
        population.append(ind)
    return population


# ========= 突然変異（可変長 rand1） =========
def mutate_rand1_varlen(population, F, current_idx, K_max, tlen, lim):
    """v = x1 + F*(x2 - x3) をスロット単位で計算し、長さは x1（基底）に合わせる。
    さらにランダムで -1ペア / そのまま / +1ペア の長さ突然変異を加える。"""
    idxs = [i for i in range(len(population)) if i != current_idx]
    r1, r2, r3 = random.sample(idxs, 3)
    x1, x2, x3 = population[r1], population[r2], population[r3]

    # x2 - x3: スロット単位の差分。足りないスロットは「相手の位置・振幅0」で補い、
    # 位置差を0にする（無かった所に乱数位置を置かない）。
    L = max(len(x2), len(x3))
    diff = []
    for i in range(L):
        p2 = x2[i] if i < len(x2) else [x3[i][0], 0.0]   # x2に無ければ x3の位置・amp0
        p3 = x3[i] if i < len(x3) else [x2[i][0], 0.0]   # x3に無ければ x2の位置・amp0
        diff.append([p2[0] - p3[0], p2[1] - p3[1]])

    # 長さは x1 に合わせる → diff を len(x1) に整列（伸長は [0,0] の no-op）して足し込む
    k1 = len(x1)
    diff_a = align_diff(diff, k1)
    v = [[x1[i][0] + F * diff_a[i][0], x1[i][1] + F * diff_a[i][1]] for i in range(k1)]

    # 長さの突然変異: ランダムで -1ペア / そのまま / +1ペア
    choice = random.randint(0, 2)
    if choice == 0 and len(v) > 1:
        v = shrink_pairs_by_amp(v, len(v) - 1)                       # 小振幅順に1ペア削る
    elif choice == 2 and len(v) < K_max:
        v.append([random.randint(0, tlen - 1), random.uniform(-lim, lim)])  # 実効振幅の新規ペア

    return clip_pairs(v, tlen, lim)


# ========= 交叉（可変長 exponential） =========
def crossover_exponential_varlen(target, mutant, CR):
    """exp 交叉（pair単位）。長さは mutant に合わせる。ターゲットを土台にし、ターゲットに無い
    スロットは mutant から取る（phantom を作らない）。連続ブロックを mutant で上書き。"""
    n = len(mutant)
    trial = [list(target[i]) if i < len(target) else list(mutant[i]) for i in range(n)]
    start = random.randint(0, n - 1)
    Lblock = 1
    while random.random() < CR and Lblock < n:
        Lblock += 1
    for i in range(Lblock):
        idx = (start + i) % n
        trial[idx] = list(mutant[idx])
    return trial


# ========= 駆動ループ（バッチ評価・greedy選択） =========
def differential_evolution_varlen(fitness_batch, population, K_max, tlen, lim,
                                  generations, F_range=(0.5, 1.0), CR=0.7,
                                  callback=None, seed=None):
    """
    fitness_batch: population(list of individuals) -> np.ndarray(評価値, 小さいほど良い)。
                   集団全体を1バッチで評価する（GPU一括推論）。
    戻り値: (best_individual, best_fitness, nit)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pop = [[list(p) for p in ind] for ind in population]
    fitness = np.asarray(fitness_batch(pop), dtype=np.float64)
    nit = 0

    for gen in range(generations):
        F = random.uniform(F_range[0], F_range[1])

        trials = []
        for i in range(len(pop)):
            mutant = mutate_rand1_varlen(pop, F, i, K_max, tlen, lim)
            trial = crossover_exponential_varlen(pop[i], mutant, CR)
            trials.append(trial)

        trial_fitness = np.asarray(fitness_batch(trials), dtype=np.float64)   # 1バッチ評価

        # greedy 選択
        for i in range(len(pop)):
            if trial_fitness[i] < fitness[i]:
                pop[i] = trials[i]
                fitness[i] = trial_fitness[i]

        nit = gen + 1
        if callback is not None:
            bi = int(np.argmin(fitness))
            callback(gen, pop[bi], float(fitness[bi]))

    bi = int(np.argmin(fitness))
    return pop[bi], float(fitness[bi]), nit
