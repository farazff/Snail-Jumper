import random
from copy import deepcopy

from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players, algorithm=1):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param algorithm: choosing algorithm
        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        for_static = sorted(players, key=lambda x: x.fitness, reverse=True)
        sorted_one = []
        if algorithm == 1:
            sorted_one = sorted(players, key=lambda x: x.fitness, reverse=True)

        sums = 0.0
        for i in for_static:
            sums = sums + i.fitness
        avg = sums / len(for_static)
        print("best:  ", for_static[0].fitness, "     avg:  ", avg, "     worst:  ", for_static[-1].fitness)

        if algorithm == 2:
            for i in range(num_players):
                wheel = random.randint(0, sums)
                tmp = 0
                for j in range(0, len(players)):
                    if wheel >= tmp:
                        tmp = tmp + players[j].fitness
                        if wheel <= tmp:
                            sorted_one.append(players[j])
                            break

        return sorted_one[: num_players]

    def generate_new_population(self, num_players, prev_players=None, algorithm=2):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param algorithm:
        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            next_gen = []
            if algorithm == 1:
                for i in range(0, len(prev_players), 2):
                    if len(next_gen) >= num_players:
                        break
                    p1 = prev_players[i]
                    p2 = prev_players[i + 1]
                    child1, child2 = self.cross_over(p1, p2)
                    child1.mutate(0.3)
                    child2.mutate(0.3)
                    next_gen.append(self.clone_player(child1))
                    next_gen.append(self.clone_player(child2))

            if algorithm == 2:
                p1 = None
                p2 = None
                sums = 0.0
                for i in prev_players:
                    sums = sums + i.fitness
                for i in range(num_players):
                    for _ in range(2):
                        wheel = random.randint(0, sums)
                        tmp = 0
                        for j in range(0, len(prev_players)):
                            if wheel >= tmp:
                                tmp = tmp + prev_players[j].fitness
                                if wheel <= tmp:
                                    if p1 is None:
                                        p1 = prev_players[j]
                                        break
                                    else:
                                        p2 = prev_players[j]

                    child1, child2 = self.cross_over(p1, p2)
                    child1.mutate(0.2)
                    child2.mutate(0.2)
                    next_gen.append(self.clone_player(child1))
                    next_gen.append(self.clone_player(child2))

            return next_gen

    def cross_over(self, p1: Player, p2: Player) -> (Player, Player):
        player1, player2 = self.clone_player(p1), self.clone_player(p2)

        for layer_num in range(len(p1.nn.weights)):
            cross_over = random.random()
            if cross_over <= 0.5:
                point = random.randint(0, p1.nn.weights[layer_num].shape[1])
                p1.cross_over(p2, layer_num, point)

        return player1, player2

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
