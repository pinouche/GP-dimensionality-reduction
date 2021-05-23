import numpy as np
from copy import deepcopy
from numpy.random import randint


def tournament(population, how_many_to_select, multi_objective, tournament_size=4):
	pop_size = len(population)
	selection = []

	if not multi_objective:
		main_objective = []
		penalty = []
		for ind in population:
			main_objective.append(ind.objectives[0][0])
			penalty.append(ind.objectives[1])

		main_objective = (main_objective - np.mean(main_objective)) / np.std(main_objective)
		penalty = (penalty - np.mean(penalty)) / np.std(penalty)

		penalized_fitness = main_objective + (1/3)*penalty

	while len(selection) < how_many_to_select:
		best_index = randint(pop_size)
		contestant_index = randint(pop_size)

		best = population[best_index]
		for i in range(tournament_size - 1):
			contestant = population[contestant_index]

			if multi_objective:
				if (contestant.rank < best.rank) or (contestant.rank == best.rank and contestant.crowding_distance > best.crowding_distance):
					best = contestant
			else:
				if penalized_fitness[contestant_index] < penalized_fitness[best_index]:
					best = contestant

		survivor = deepcopy(best)
		selection.append(survivor)

	return selection
