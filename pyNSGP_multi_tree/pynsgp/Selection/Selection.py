import numpy as np
from copy import deepcopy
from numpy.random import randint


def tournament(population, how_many_to_select, multi_objective, tournament_size=4):
	pop_size = len(population)
	selection = []

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
				if contestant.objectives[0][0] < best.objectives[0][0]:
					best = contestant

		survivor = deepcopy(best)
		selection.append(survivor)

	return selection
