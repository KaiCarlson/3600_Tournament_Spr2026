from collections.abc import Callable
from typing import List, Set, Tuple
import random
import numpy as np

from game import board, move, enums

class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        self.T = np.asarray(transition_matrix, dtype=np.float64)
        self.n = 64

        # Normalize just in case
        row_sums = self.T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.T = self.T / row_sums

        # Initial prior: rat starts at (0,0), then moves 1000 times
        self.belief = np.zeros(self.n, dtype=np.float64)
        self.belief[0] = 1.0
        for _ in range(1000):
            self.belief = self.belief @ self.T

        self.dist_table = self._precompute_distances()
        self.last_opponent_search_seen = None

        # Observation models from rat.py / assignment
        self.noise_probs = {
            enums.Cell.BLOCKED: np.array([0.5, 0.3, 0.2], dtype=np.float64),
            enums.Cell.SPACE:   np.array([0.7, 0.15, 0.15], dtype=np.float64),
            enums.Cell.PRIMED:  np.array([0.1, 0.8, 0.1], dtype=np.float64),
            enums.Cell.CARPET:  np.array([0.1, 0.1, 0.8], dtype=np.float64),
        }
        self.dist_offsets = (-1, 0, 1, 2)
        self.dist_probs = (0.12, 0.7, 0.12, 0.06)


    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        # rat probability calc
        self._incorporate_opponent_search(board)
        self.belief = self.belief @ self.T
        noise, est_dist = sensor_data
        self._update_belief_from_sensor(board, noise, est_dist)
        best_idx = int(np.argmax(self.belief))
        best_loc = self._index_to_loc(best_idx)
        best_p = float(self.belief[best_idx])

        if best_p > 0.5:
            return move.Move.search(best_loc)

        valid_moves = board.get_valid_moves(exclude_search=True)
        if not valid_moves:
            return move.Move.search(best_loc)
        best_move = None
        best_score = -float("inf")
        for mv in valid_moves:
            score = self._score_move(board, mv)
            if score > best_score:
                best_score = score
                best_move = mv

        return best_move if best_move is not None else valid_moves[0]

    def _incorporate_opponent_search(self, board):
        loc, success = board.opponent_search

        tag = (loc, success)
        if loc is None or tag == self.last_opponent_search_seen:
            return

        self.last_opponent_search_seen = tag

        idx = self._loc_to_index(loc)

        if success:
            # New rat spawned and took 1000 hidden moves before next turn
            self.belief = np.zeros(self.n, dtype=np.float64)
            self.belief[0] = 1.0
            for _ in range(1000):
                self.belief = self.belief @ self.T
        else:
            # Rat was not at searched cell at search time.
            # Approximation: zero it out now, then renormalize.
            self.belief[idx] = 0.0
            self._renormalize()

    def _precompute_distances(self):
        table = {}
        cells = [(x, y) for y in range(8) for x in range(8)]
        for a in cells:
            inner = {}
            for b in cells:
                inner[b] = abs(a[0] - b[0]) + abs(a[1] - b[1])
            table[a] = inner
        return table

    def _update_belief_from_sensor(self, board, noise, est_dist):
        my_pos = board.player_worker.get_location()
        new_belief = np.zeros(64, dtype=np.float64)

        for idx in range(64):
            rat_loc = self._index_to_loc(idx)

            # Likelihood of the reported noise at this square
            cell_type = board.get_cell(rat_loc)
            noise_like = self.noise_probs[cell_type][int(noise)]

            # Likelihood of the reported distance if rat were here
            actual_d = self._manhattan(my_pos, rat_loc)
            dist_like = self._distance_likelihood(actual_d, est_dist)

            # Bayesian reweighting
            new_belief[idx] = self.belief[idx] * noise_like * dist_like

        total = new_belief.sum()
        if total <= 0:
            # fallback if numerical underflow / impossible observation
            self.belief[:] = 1.0 / 64.0
        else:
            self.belief = new_belief / total

    def _distance_likelihood(self, actual_d, observed_d):
        prob = 0.0
        for off, p in zip(self.dist_offsets, self.dist_probs):
            reported = max(0, actual_d + off)
            if reported == observed_d:
                prob += p
        return prob

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _index_to_loc(self, idx):
        return (idx % 8, idx // 8)

    def _loc_to_index(self, pos):
        x, y = pos
        return y * 8 + x

    def _renormalize(self):
        s = self.belief.sum()
        if s <= 0:
            self.belief[:] = 1.0 / 64.0
        else:
            self.belief /= s

    def _score_move(self, board, mv):
        next_board = board.forecast_move(mv, check_ok=True)
        if next_board is None:
            return -1e18

        score = 0.0

        me_now = board.player_worker.get_points()
        me_next = next_board.player_worker.get_points()

        # Immediate score gain
        score += 20.0 * (me_next - me_now)

        # Position after move
        next_pos = next_board.player_worker.get_location()

        # Get closer to likely rat cells
        rat_term = 0.0
        for idx, p in enumerate(self.belief):
            if p < 1e-6:
                continue
            pos = self._index_to_loc(idx)
            rat_term -= p * self.dist_table[next_pos][pos]
        score += 0.2 * rat_term

        # Favor carpet opportunities created for future turns
        score += 3.0 * self._future_carpet_potential(next_board)

        # Prefer denying opponent access a little
        score -= 1.5 * self._opponent_future_potential(next_board)

        return score

    def _future_carpet_potential(self, board):
        total = 0.0
        for mv in board.get_valid_moves(exclude_search=True):
            if mv.move_type == enums.MoveType.CARPET:
                # nonlinear carpet rewards matter a lot
                total += [0, -1, 2, 4, 6, 10, 15, 21][mv.roll_length]
        return total

    def _opponent_future_potential(self, board):
        sim = board.get_copy()
        sim.reverse_perspective()
        total = 0.0
        for mv in sim.get_valid_moves(exclude_search=True):
            if mv.move_type == enums.MoveType.CARPET:
                total += [0, -1, 2, 4, 6, 10, 15, 21][mv.roll_length]
        return total

