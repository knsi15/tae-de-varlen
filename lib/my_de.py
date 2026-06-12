import numpy as np
import random
from scipy.optimize import minimize

# --- Core Components ---

def _ensure_bounds(vec, bounds):
    """Ensure that a vector respects the given bounds."""
    vec_bounded = np.copy(vec)
    # Clip the values to be within the bounds
    for i, (min_val, max_val) in enumerate(bounds):
        if vec_bounded[i] < min_val:
            vec_bounded[i] = min_val
        elif vec_bounded[i] > max_val:
            vec_bounded[i] = max_val
    return vec_bounded

def _objective_function_wrapper(func, args=()):
    """Wrapper to pass additional arguments to the objective function."""
    return lambda x: func(x, *args)

# --- Initialization ---

def _init_population_random(popsize, dimensions, bounds, seed=None):
    """Initializes the population with random values within the bounds."""
    if seed is not None:
        np.random.seed(seed)
    
    population = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + population * diff
    return population

# --- Main DE Algorithm ---

def differential_evolution(
    func,
    bounds,
    args=(),
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    tol=0.01,
    mutation=0.5,
    recombination=0.7,
    seed=None,
    polish=True,
    updating='immediate',
    patience=10
):
    """
    A corrected and simplified implementation of Differential Evolution.

    This version addresses key issues from the original implementation:
    1. Boundary constraints are now strictly enforced.
    2. The evolution loop correctly processes the entire population.
    3. Population initialization is generalized.
    4. An optional polishing step is included.
    5. Supports 'immediate' and 'deferred' updating.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Prepare objective function
    objective_func = _objective_function_wrapper(func, args)

    # Setup population
    dimensions = len(bounds)
    population_size = popsize * dimensions
    population = _init_population_random(population_size, dimensions, bounds, seed)
    
    fitness = np.asarray([objective_func(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_vector = population[best_idx]
    
    no_improve_count = 0

    # --- Evolution Loop ---
    for generation in range(maxiter):
        
        if updating == 'deferred':
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

        for i in range(population_size):
            # --- Mutation ---
            # Select 3 random individuals (r1, r2, r3) different from current one
            idxs = [idx for idx in range(population_size) if idx != i]
            r1, r2, r3 = random.sample(idxs, 3)
            
            x_r1 = population[r1]
            x_r2 = population[r2]
            x_r3 = population[r3]

            # For simplicity, only 'rand1' strategy is implemented here.
            # More strategies can be added as needed.
            # Example: mutant = best_vector + mutation * (x_r1 - x_r2) for 'best1'
            mutant = x_r1 + mutation * (x_r2 - x_r3)
            
            # Enforce boundary constraints
            mutant = _ensure_bounds(mutant, bounds)

            # --- Crossover (Binomial) ---
            cross_points = np.random.rand(dimensions) < recombination
            # Ensure at least one parameter from mutant is chosen
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # --- Selection ---
            trial_fitness = objective_func(trial)

            if trial_fitness < fitness[i]:
                if updating == 'immediate':
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_vector = trial
                elif updating == 'deferred':
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
        
        if updating == 'deferred':
            population = new_population
            fitness = new_fitness
            best_idx = np.argmin(fitness)
            current_best_fitness = fitness[best_idx]
        else: # immediate
            current_best_fitness = best_fitness

        print(f"Generation {generation + 1}: Best fitness = {current_best_fitness}")

        # --- Convergence Check ---
        if abs(current_best_fitness - best_fitness) < tol:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Convergence criteria met. Stopping after {generation + 1} generations.")
                break
        else:
            no_improve_count = 0
            best_fitness = current_best_fitness
            best_vector = population[np.argmin(fitness)]

    # --- Polishing Step ---
    if polish:
        print("Polishing the final solution...")
        result = minimize(
            objective_func,
            best_vector,
            method='L-BFGS-B',
            bounds=bounds
        )
        if result.fun < best_fitness:
            best_fitness = result.fun
            best_vector = result.x
            print("Polishing improved the solution.")

    return {
        'x': best_vector,
        'fun': best_fitness,
        'message': 'Optimization terminated successfully.',
        'success': True
    }
