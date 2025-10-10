import random

def evaluate_path(path, packages):
    total_distance = 0
    for i in range(len(path) - 1):
        x1, y1 = packages[path[i]]
        x2, y2 = packages[path[i+1]]
        total_distance += abs(x1 - x2) + abs(y1 - y2)
    return total_distance

def best_route(package_list, population_size=50, mut_prob=0.2, generations=50):
    n = len(package_list)
    population = [random.sample(range(n), n) for _ in range(population_size)]

    for _ in range(generations):
        population.sort(key=lambda path: evaluate_path(path, package_list))
        new_population = population[:10]  

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:25], 2)
            cut = random.randint(1, n - 1)
            child = parent1[:cut] + [p for p in parent2 if p not in parent1[:cut]]

            if random.random() < mut_prob:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]

            new_population.append(child)

        population = new_population

    best_solution = population[0]
    print('best route: ', best_solution)
    return best_solution
