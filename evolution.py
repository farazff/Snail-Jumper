import copy
import random

from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """

        sortedOne = sorted(players, key=lambda x: x.fitness, reverse=True)

        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)
        return sortedOne[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            next_gen = []
            for i in range(0, len(prev_players), 2):
                if len(next_gen) >= num_players:
                    break
                p1 = prev_players[i]
                p2 = prev_players[i + 1]
                child1, child2 = self.cross_over(p1, p2)
                child1.mutate(0.1)
                child2.mutate(0.1)
                next_gen.append(self.clone_player(child1))
                next_gen.append(self.clone_player(child2))

            return next_gen

    def cross_over(self, p1: Player, p2: Player) -> (Player, Player):
        player1, player2 = self.clone_player(p1), self.clone_player(p2)

        for layer_num in range(len(p1.nn.weights)):
            cross_over = random.uniform(0, 1)
            if cross_over <= 0.5:
                point = random.randint(0, p1.nn.weights[layer_num].shape[1])
                p1.cross_over(p2, layer_num, point)

        return player1, player2

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
