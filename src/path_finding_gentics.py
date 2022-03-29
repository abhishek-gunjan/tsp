import numpy as np
import random
import math
import matplotlib.pyplot as plt

'''
Genetical path finding
Finds locally best ways from L service centers with [M0, M1, ..., ML] engineers
through atms_number ATMs and back to their service center
'''
def fitness_pop(population):
    fitness_result = np.zeros(len(population))
    for i in range(len(fitness_result)):
        fitness_result[i] = fitness(population[i])
    return fitness_result

def fitness(creature):
    sum_dist = np.zeros(len(creature))
    for j in range(len(creature)):
        mat_path = np.zeros((dist.shape[0], dist.shape[1]))
        path = creature[j]
        if len(path) != 0:
            for v in range(len(path)):
                if v == 0:
                    mat_path[engineers[j], path[v]] = 1
                else:
                    mat_path[path[v - 1] + service_centers, path[v]] = 1
            mat_path = mat_path * dist
            sum_dist[j] = (np.sum(mat_path) + dist[engineers[j], path[-1]]) / velocity + repair_time * len(path)
    return np.max(sum_dist)

def birth_prob(fitness_result):
    birth_prob = np.abs(fitness_result - np.max(fitness_result))
    birth_prob = birth_prob / np.sum(birth_prob)
    return birth_prob

def mutate(creat, engi):
    pnt_1 = random.randint(0, len(creat)-1)
    pnt_2 = random.randint(0, len(creat)-1)
    if random.random() < mut_1_prob:
        creat[pnt_1], creat[pnt_2] = creat[pnt_2], creat[pnt_1]
    if random.random() < mut_2_prob and pnt_1 != pnt_2:
        if pnt_1 > pnt_2:
            pnt_1, pnt_2 = pnt_2, pnt_1
        creat[pnt_1:pnt_2+1] = list(reversed(creat[pnt_1:pnt_2+1]))
    if random.random() < mut_3_prob:
        engi = [number-1 for number in engi if number != 0]
        # engi = [number - 2 for number in engi if number > 1]
        while(sum(engi) != atms_number):
            engi[random.randint(0, len(engi)-1)] += 1
    return creat, engi

def two_opt(creature):
    sum_dist = np.zeros(len(creature))
    for j in range(len(creature)):
        mat_path = np.zeros((dist.shape[0], dist.shape[1]))
        path = creature[j]
        if len(path) != 0:
            for v in range(len(path)):
                if v == 0:
                    mat_path[engineers[j], path[v]] = 1
                else:
                    mat_path[path[v - 1] + service_centers, path[v]] = 1
            mat_path = mat_path * dist
            sum_dist[j] = (np.sum(mat_path) + dist[engineers[j], path[-1]]) / velocity + repair_time * len(path)
    for u in range(len(creature)):
        best_path = creature[u].copy()
        while True:
            previous_best_path = best_path.copy()
            for x in range(len(creature[u])-1):
                for y in range(x + 1, len(creature[u])):
                    path = best_path.copy()
                    if len(path) != 0:
                        path = path[:x] + list(reversed(path[x:y])) + path[y:]      # 2-opt swap
                        mat_path = np.zeros((dist.shape[0], dist.shape[1]))
                        for v in range(len(path)):
                            if v == 0:
                                mat_path[engineers[u], path[v]] = 1
                            else:
                                mat_path[path[v - 1] + service_centers, path[v]] = 1
                        mat_path = mat_path * dist
                        sum_dist_path = (np.sum(mat_path) + dist[engineers[u], path[-1]]) / velocity + repair_time * len(path)
                        if sum_dist_path < sum_dist[u]:
                            best_path = path.copy()
                            creature[u] = path.copy()
            if previous_best_path == best_path:
                break
    return creature

def crossover_mutation(population, birth_prob):
    new_population = []
    for i in range(round(len(population)/2)):
        prob = np.random.rand(birth_prob.size) - birth_prob
        pair = np.zeros(2).astype(int)
        pair[0] = np.argmin(prob)
        pair[1] = random.randint(0, prob.size-1)
        engi_1 = [len(population[pair[0]][v]) for v in range(len(population[pair[0]]))]
        engi_2 = [len(population[pair[1]][v]) for v in range(len(population[pair[1]]))]
        parent_1 = []
        parent_2 = []
        for j in range(len(engi_1)):
            parent_1 += population[pair[0]][j]
        for j in range(len(engi_2)):
            parent_2 += population[pair[1]][j]
        creat_1 = [-1] * len(parent_1)
        creat_2 = [-1] * len(parent_2)
        cross_point_1 = random.randint(0, len(parent_1) - 1)
        cross_point_2 = random.randint(0, len(parent_2) - 1)
        node_1 = parent_1[cross_point_1:]
        node_2 = parent_2[cross_point_2:]
        w = 0
        for v in range(len(creat_1)):
            if parent_2[v] not in node_1:
                creat_1[v] = parent_2[v]
            else:
                creat_1[v] = node_1[w]
                w += 1
        w = 0
        for v in range(len(creat_2)):
            if parent_1[v] not in node_2:
                creat_2[v] = parent_1[v]
            else:
                creat_2[v] = node_2[w]
                w += 1
        # mutations
        creat_1, engi_1 = mutate(creat_1, engi_1)
        creat_2, engi_2 = mutate(creat_2, engi_2)
        # children
        child_1 = []
        engi_sum = 0
        for v in range(len(engi_1)):
            child_1.append(creat_1[engi_sum:engi_sum+engi_1[v]])
            engi_sum += engi_1[v]
        child_2 = []
        engi_sum = 0
        for v in range(len(engi_2)):
            child_2.append(creat_2[engi_sum:engi_sum + engi_2[v]])
            engi_sum += engi_2[v]
        together = [child_1, child_2, population[pair[0]], population[pair[1]]]
        fit = np.array([fitness(creature) for creature in together])
        fit = fit.argsort()
        if two_opt_search:
            new_population.append(two_opt(together[fit[0]]))
            new_population.append(two_opt(together[fit[1]]))
        else:
            new_population.append(together[fit[0]])
            new_population.append(together[fit[1]])
    return new_population

def plot_paths(paths):
    plt.clf()
    plt.title('Best path overall')
    for v in range(service_centers):
        plt.scatter(points_locations[v, 0], points_locations[v, 1], c='r')
    for v in range(atms_number):
        plt.scatter(points_locations[v+service_centers, 0], points_locations[v+service_centers, 1], c='b')
    for v in range(len(paths)):
        if len(paths[v]) != 0:
            path_locations = points_locations[service_centers:]
            path_locations = path_locations[np.array(paths[v])]
            path_locations = np.vstack((points_locations[engineers[v]], path_locations))
            path_locations = np.vstack((path_locations, points_locations[engineers[v]]))
            plt.plot(path_locations[:, 0], path_locations[:, 1])
    plt.show()
    plt.pause(0.0001)

# Bank parameters
atms_number = 50         # ATM quantity
service_centers = 3     # service centers quantity
velocity = 100             # 100 / hour
repair_time = 0         # 0.5 hour
max_engi = 3              # maximum number of engineers in one service center

# genetic parameters
population_size = 50    # population size (even number!)
generations = 1000       # population's generations
mut_1_prob = 0.4         # prob of replacing together two atms in combined path
mut_2_prob = 0.6      # prob of reversing the sublist in combined path
mut_3_prob = 0.8     # probability of changing the length of paths for engineers
two_opt_search = False  # better convergence, lower speed for large quantity of atms


# seed
np.random.seed(2)
random.seed(1)
plt.ion()
engineers = []



"""
Below code of getting engineers 

e.g - array([0, 1, 1, 1, 2])

"""
for i in range(service_centers):
    for j in range(random.randint(1, max_engi)):
        engineers.append(i)
engineers = np.array(engineers)
print('Engineers: {}'.format(engineers))

"""
Below code where distance is getting calculate 

e.g - array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])

"""

dist = np.zeros((atms_number+service_centers, atms_number))
points_locations = np.random.randint(0, 100, (service_centers+atms_number)*2)

"""
Below line to calculate points locations 
array([[40, 15],
       [72, 22],
       [43, 82],
       [75,  7],
       [34, 49],
       [95, 75],
       [85, 47],
       [63, 31],
       [90, 20],
       [37, 39],
       [67,  4],
       [42, 51],
       [38, 33],
       [58, 67],
       [69, 88],
       [68, 46],
       [70, 95],
       [83, 31],
       [66, 80],
       [52, 76],
       [50,  4],
       [90, 63],
       [79, 49],
       [39, 46],
       [ 8, 50],
       [15,  8],
       [17, 22],
       [73, 57],
       [90, 62],
       [83, 96],
       [43, 32],
       [26,  8],
       [76, 10],
       [40, 34],
       [60,  9],
       [70, 86],
       [70, 19],
       [56, 82],
       [ 1, 68],
       [40, 81],
       [61, 70],
       [97, 18],
       [84, 90],
       [87, 22],
       [43, 52],
       [74, 72],
       [90, 99],
       [91, 96],
       [16, 55],
       [21, 43],
       [93, 80],
       [40, 70],
       [74, 37]])

"""
points_locations = points_locations.reshape((service_centers+atms_number, 2))


for i in range(dist.shape[0]):
    for j in range(dist.shape[1]):
        dist[i, j] = math.sqrt((points_locations[i, 0] - points_locations[j + service_centers, 0]) ** 2 +
                               (points_locations[i, 1] - points_locations[j + service_centers, 1]) ** 2)
        if j+service_centers == i:
            dist[i][j] = 0
# random population creation

"""
sample population array 
[[[7, 32, 29, 33, 45, 25, 14, 6, 38], [28], [48, 0, 37, 21, 18, 9, 30, 2, 3, 1, 36, 31, 4, 23], [26, 47, 5, 40, 17, 39, 42], [46, 19, 34, 20, 35, 12, 49, 24, 15, 8
, 41, 11, 13, 44, 22, 10, 43, 27, 16]], [[42, 12, 20, 19, 40, 34, 36, 28, 45, 2, 37, 17, 31, 33, 13, 30, 32], [44, 7, 39],....
"""
population = []
for i in range(population_size):
    atms_range = list(range(atms_number))
    pop = [0] * engineers.size
    for j in range(engineers.size):
        pop[j] = []
        if len(atms_range) != 0:
            if j != engineers.size-1:
                for v in range(random.randint(1, round(2*atms_number/engineers.size))):
                    pop[j].append(random.choice(atms_range))
                    atms_range.remove(pop[j][-1])
                    if len(atms_range) == 0:
                        break
            else:
                for v in range(len(atms_range)):
                    pop[j].append(random.choice(atms_range))
                    atms_range.remove(pop[j][-1])
    population.append(pop)

import pdb
pdb.set_trace()
"""
fitness_result
array([ 8.3610049 , 12.18254411,  8.18962019,  9.09406337,  9.83132276,
        9.49915485, 12.16496874,  9.77793808,  8.93768375,  6.92068589,
       11.17080457, 11.01275734, 11.41729378,  9.48986852,  8.53881809,
        5.9080962 , 10.59011395,  8.63676556, 10.45126391,  9.56184227,....

"""
fitness_result = fitness_pop(population)
best_mean_creature_result = np.mean(fitness_result)
best_creature_result = np.min(fitness_result)
best_selection_prob = birth_prob(fitness_result)
selection_prob = best_selection_prob
new_population = population.copy()
plot_paths(population[np.argmin(fitness_result)])
for i in range(generations):
    new_population = crossover_mutation(population, selection_prob)
    fitness_result = fitness_pop(new_population)
    mean_creature_result = np.mean(fitness_result)
    if np.min(fitness_result) < best_creature_result:
        plot_paths(population[np.argmin(fitness_result)])
        best_creature_result = np.min(fitness_result)
    if mean_creature_result < best_mean_creature_result:
        best_mean_creature_result = mean_creature_result
        best_selection_prob = birth_prob(fitness_result)
        selection_prob = best_selection_prob
        population = new_population.copy()
    print('Mean population time: {0} Best time: {1}'.format(best_mean_creature_result, best_creature_result))
plt.ioff()
plt.show()