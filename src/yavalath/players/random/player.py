import random

from yavalath.core.player import Player


class RandomPlayer(Player):
    def calc_best(self, board, player):
        empty_cells = board.get_empty_cells()
        return random.choice(empty_cells)
