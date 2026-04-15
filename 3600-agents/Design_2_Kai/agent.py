import numpy as np
from game.enums import MoveType
from .rat_belief import RatBelief
from .search import best_move


class PlayerAgent:
    """
    Thin wiring layer. All logic lives in rat_belief.py, heuristic.py, search.py.

    Responsibilities:
        - Own the RatBelief instance across turns
        - Update belief in the correct order each turn
        - Delegate move selection to search.best_move()
        - Track last move for own-search feedback
    """

    def __init__(self, board, transition_matrix, time_left_func):
        """
        Called once before the game starts.

        Parameters
        ----------
        board : Board
            Initial board state (workers already placed, no moves made).
        transition_matrix : array-like, shape (64, 64)
            Rat transition matrix. May be a JAX array — RatBelief converts to numpy.
        time_left_func : callable
            Returns remaining constructor time. Not used here but required by interface.
        """
        self._belief = RatBelief(transition_matrix)

        # Track the move we made last turn so we can apply the own-search
        # hard update at the start of the next turn. The engine delivers our
        # own search result via board.player_search, but only one turn later.
        self._last_move = None

    def play(self, board, rat_samples, time_left_func):
        """
        Called every turn. Returns a Move object.

        Belief update order (all four steps must happen before deciding):
            1. predict()                 — rat already moved before this call
            2. update()                  — incorporate this turn's observation
            3. observe_opponent_search() — hard update from opponent's last action
            4. observe_own_search()      — hard update from our OWN last turn's result
                                           (delivered back to us via board.player_search)

        Parameters
        ----------
        board : Board
            Current board state from the engine's perspective (player_worker = us).
        rat_samples : tuple[Noise, int]
            (noise_type, noisy_distance) from rat.sample().
        time_left_func : callable
            Returns remaining turn time in seconds when called.

        Returns
        -------
        Move
            The selected move.
        """
        noise, dist = rat_samples
        worker_pos = board.player_worker.get_location()

        # --- Step 1: Predict ---
        # Rat has already moved this turn before we were called.
        self._belief.predict()

        # --- Step 2: Update from observation ---
        self._belief.update(noise, dist, worker_pos, board)

        # --- Step 3: Hard update from opponent's search ---
        # board.opponent_search = (loc, found) or (None, False) if they didn't search.
        opp_loc, opp_found = board.opponent_search
        self._belief.observe_opponent_search(opp_loc, opp_found)

        # --- Step 4: Hard update from our OWN last search result ---
        # The engine delivers our last search result via board.player_search.
        # This is (None, False) if we didn't search last turn.
        own_loc, own_found = board.player_search
        if own_loc is not None:
            self._belief.observe_own_search(own_loc, own_found)

        # --- Decide and return move ---
        move = best_move(board, self._belief, time_left_func)
        self._last_move = move
        return move

    def commentate(self):
        """
        Called once after the game ends. Return a debug string.
        Useful for post-game diagnostics — print belief stats, turn counts, etc.
        """
        belief = self._belief.get_belief()
        best_i = int(np.argmax(belief))
        best_p = float(belief[best_i])
        best_loc = (best_i % 8, best_i // 8)
        ev = 6.0 * best_p - 2.0

        return (
            f"Final belief peak: {best_p:.3f} at {best_loc}, "
            f"search EV={ev:.3f}"
        )