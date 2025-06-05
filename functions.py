import numpy as np
import json
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from individual import Individual, rng
from config import GENOME_LENGTH, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, NO_PROGRESS_LIMIT, RESULTS_DIR

def crossover(parent1, parent2):
    num_splits = int(np.floor(np.log2(GENOME_LENGTH)))
    points = sorted(rng.choice(range(1, GENOME_LENGTH), num_splits, replace=False))
    points = [0] + points + [GENOME_LENGTH]

    new_genome = np.empty(GENOME_LENGTH, dtype=np.float32)
    toggle = True
    for i in range(len(points) - 1):
        s, e = points[i], points[i + 1]
        new_genome[s:e] = parent1.genome[s:e] if toggle else parent2.genome[s:e]
        toggle = not toggle

    return Individual(new_genome)


# def mutate(individual, generation):
#     progress = generation / GENERATIONS
#     mutation_rate = MUTATION_RATE * (1 - progress) # symulwanym wy..a..aniem
#     sigma = 0.5 * (1 - progress) + 0.01 * progress # halo, halo?
#     for i in range(GENOME_LENGTH):
#         if rng.random() < mutation_rate:
#             individual.genome[i] += rng.normal(0, sigma)
#             individual.genome[i] = np.clip(individual.genome[i], -1, 1)
#     return individual

def mutate(individual, sigma=0.05):
    for i in range(GENOME_LENGTH):
        if rng.random() < MUTATION_RATE:
            #scale = rng.uniform(0.00001, 2.0)
            individual.genome[i] += rng.normal(0, sigma) #* scale
            individual.genome[i] = np.clip(individual.genome[i], -1, 1)
    return individual




# def roulette_wheel_selection(population):

#     fitnesses = np.array([ind.fitness for ind in population])
#     min_fitness = np.min(fitnesses)
#     if min_fitness < 0:
#         fitnesses = fitnesses - min_fitness + 1e-6 #tzw. magiczne zmienne 
#     total_fitness = np.sum(fitnesses)
#     probs = fitnesses / total_fitness
#     selected_index = rng.choice(len(population), p=probs)
#     return population[selected_index]

# def build_roulette(population):
#     roulette = []
#     for ind in population:
#         count = max(int(ind.fitness), 0)  # zabezpieczenie na fitness <= 0
#         roulette.extend([ind] * count)
#     return roulette if roulette else population  # fallback, jeśli pusta

# def build_roulette(population):
#     min_fitness = min(ind.fitness for ind in population)
#     offset = 1 - min_fitness if min_fitness < 1 else 0  # tak, żeby wszystko było ≥1 

#     roulette = []
#     for ind in population:
#         weight = int(ind.fitness + offset)
#         if weight > 0:
#             roulette.extend([ind] * weight)

#     return roulette if roulette else population

def build_roulette(population):
    fitnesses = np.array([ind.fitness for ind in population])
    min_fitness = np.min(fitnesses)

    if min_fitness < 0:
        fitnesses = fitnesses - min_fitness + 1  # przesunięcie tak, by wszystko było dodatnie      ----> najlepsze wyniki, ale skacze między - a +

    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:
        return population  # fallback

    probs = fitnesses / total_fitness
    # losuj wskaźniki zgodnie z prawdopodobieństwami (może być z powtórzeniami)
    indices = rng.choice(len(population), size=POPULATION_SIZE, p=probs, replace=True)
    return [population[i] for i in indices]

# def build_roulette(population):
#     fitnesses = np.array([ind.fitness for ind in population])

#     if np.all(fitnesses <= 0):  # Jeśli wszystkie wartości są ujemne lub zero
#         scaled_fitness = np.exp(fitnesses)  # Użyj funkcji wykładniczej
#     else:
#         # Przesuń wszystkie wartości tak, aby minimum było równe 1
#         scaled_fitness = fitnesses - np.min(fitnesses) + 1

#     total = np.sum(scaled_fitness)
#     if total == 0:
#         # Awaryjne zabezpieczenie - równe prawdopodobieństwo
#         probs = np.ones(len(population)) / len(population)
#     else:
#         probs = scaled_fitness / total 

#     selected_index = rng.choice(len(population), p=probs)
#     return population[selected_index]


# def build_roulette(population):
#     # Tworzymy listę ruletki z powtórzonymi osobnikami według ich fitness
#     ruletka = []
#     for ind in population:
#         # Zakładamy, że fitness jest nieujemny i im większy, tym lepszy
#         # Liczba powtórzeń = fitness zaokrąglony do całkowitej
#         repetitions = max(1, int(round(ind.fitness)))
#         ruletka.extend([ind] * repetitions)


#     if not ruletka:  # zabezpieczenie przed pustą ruletką
#         return rng.choice(population)

#     rodzic = rng.choice(ruletka)
#     return rodzic

# def build_roulette(population):
#     fitnesses = np.array([ind.fitness for ind in population])

#     min_fitness = np.min(fitnesses)
#     if min_fitness < 0:
#         fitnesses = fitnesses - min_fitness + 1  # przesunięcie

#     max_fitness = np.max(fitnesses)
#     if max_fitness > 0:
#         fitnesses = (fitnesses / max_fitness) * 99 + 1

#     ruletka = []
#     for ind, fit in zip(population, fitnesses):
#         repetitions = max(1, int(round(fit)))
#         ruletka.extend([ind] * repetitions)

#     return ruletka if ruletka else population  # zabezpieczenie

# def roulette_wheel_selection(population):
#     ruletka = build_roulette(population)
#     return rng.choice(ruletka)

'''
ruletka = []
for p in populacja:
    for _ in p.fitness:
    ruletka.append(p)

# Wybieranie rodziców do krzyżowania z ruletki:
for _ in range(len(populacja)):
    rodzic_1, rodzic_2 = ruletka.choice(), ruletka.choice()
    # operacja krzyżowania 
    # crossover(rodzic_1, rodzic_2)

'''


def evolve(env):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            os.remove(os.path.join(RESULTS_DIR, filename))

    population = [Individual() for _ in range(POPULATION_SIZE)]
    best_history = []
    best_score = -np.inf
    no_progress = 0
    all_generations_data = []

    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen + 1} ===")

        for individual in population:
            individual.evaluate(env)

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        current_best = population[0].fitness
        avg_fitness = np.mean([ind.fitness for ind in population])

        print(f"Best: {current_best:.2f} | Avg: {avg_fitness:.2f}")

        best_history.append((gen + 1, current_best, avg_fitness))

        # Zapisujemy całą populację do all_generations_data
        all_generations_data.append(deepcopy(population))


        if current_best > best_score:
            best_score = current_best
            best_individual = deepcopy(population[0])
            no_progress = 0
        else:
            no_progress += 1
            if no_progress >= NO_PROGRESS_LIMIT:
                print("♻️ Delikatny restart — mutujemy najlepszego osobnika do nowej populacji")
                population = [mutate(deepcopy(best_individual), gen) for _ in range(POPULATION_SIZE)]
                no_progress = 0
                continue

        next_gen = [population[0], population[1]]  # Elitaryzm

        roulette = build_roulette(population)
        if not roulette:  # zabezpieczenie przed pustą ruletką
            return rng.choice(population)

        while len(next_gen) < POPULATION_SIZE:
            p1 = rng.choice(roulette)
            p2 = rng.choice(roulette)
            # p1 = build_roulette(population)
            # p2 = build_roulette(population)
            # p1 = roulette_wheel_selection(population)
            # p2 = roulette_wheel_selection(population)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)


    # 🔽 Zapis danych generacji w nowym formacie
    for gen_idx, gen_population in enumerate(all_generations_data):
        generation_number = gen_idx + 1
        filename = os.path.join(RESULTS_DIR, f"generation_{generation_number:03d}.json")

        individuals_data = []
        for ind_id, ind in enumerate(gen_population):
            individuals_data.append({
                "id": ind_id,
                "genotyp": ind.genome.tolist(),
                "fitness": float(ind.fitness)
            })

        generation_data = {
            "nr_generacji": generation_number,
            "n_populacji": POPULATION_SIZE,
            "osobnicy": individuals_data
        }

        with open(filename, "w") as f:
            json.dump(generation_data, f, indent=4)


def plot_fitness_history(results_dir = "venv/results"):
    files = sorted(
        [f for f in os.listdir(results_dir) if f.startswith("generation_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    generations = []
    best_scores = []
    avg_scores = []

    for file in files:
        with open(os.path.join(results_dir, file), "r") as f:
            entry = json.load(f)
            generations.append(entry["nr_generacji"])

            fitnesses = [ind["fitness"] for ind in entry["osobnicy"]]
            best_scores.append(max(fitnesses))
            avg_scores.append(sum(fitnesses) / len(fitnesses))

    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_scores, label="Najlepszy fitness", linewidth=2)
    plt.plot(generations, avg_scores, label="Średni fitness", linewidth=2, linestyle="--")
    plt.xlabel("Generacja")
    plt.ylabel("Fitness")
    plt.title("Fitness w kolejnych generacjach")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_population_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    population = []
    for osobnik in data["osobnicy"]:
        genome = np.array(osobnik["genotyp"], dtype=np.float32)
        fitness = osobnik["fitness"]
        ind = Individual(genome)
        ind.fitness = fitness
        population.append(ind)

    return data["nr_generacji"], population


def load_generation(gen_number):
    filename = f"generation_{gen_number:03d}.json"  # zerowe wypełnienie do 3 cyfr
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")
    return load_population_from_json(file_path)



def run_individual_in_env(env, individual, max_steps=1000):
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    truncated = False
    step = 0

    while not (done or truncated) and step < max_steps:
        action = individual.act(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        env.render()
        step += 1

    print(f"Total reward: {total_reward:.2f}")

