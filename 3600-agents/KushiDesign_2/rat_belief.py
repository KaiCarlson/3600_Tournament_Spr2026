import numpy as np
from game.enums import Cell, Noise, BOARD_SIZE

# ---------------------------------------------------------------------------
# Observation model constants (from assignment spec)
# ---------------------------------------------------------------------------

# P(noise_type | floor_type)
# Indexed as _NOISE_EMISSION[cell_type][noise.value]
# Noise.SQUEAK=0, Noise.SCRATCH=1, Noise.SQUEAL=2
_NOISE_EMISSION = {
    Cell.BLOCKED: np.array([0.5,  0.3,  0.2]),
    Cell.SPACE:   np.array([0.7,  0.15, 0.15]),
    Cell.PRIMED:  np.array([0.1,  0.8,  0.1]),
    Cell.CARPET:  np.array([0.1,  0.1,  0.8]),
}

# P(reported_distance = actual_distance + offset) for offset in {-1, 0, +1, +2}
# Key: offset, Value: probability
_DIST_ERROR_PROBS = {
    -1: 0.12,
     0: 0.70,
    +1: 0.12,
    +2: 0.06,
}


def _cell_to_index(x: int, y: int) -> int:
    return y * BOARD_SIZE + x


def _index_to_cell(i: int) -> tuple[int, int]:
    return (i % BOARD_SIZE, i // BOARD_SIZE)


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


class RatBelief:
    """
    Hidden Markov Model belief tracker for the rat's position.

    State space: 64 cells (8x8 board), indexed as i = y*8 + x.
    Belief is a probability distribution over all 64 cells.

    Update cycle each turn (BEFORE calling this, the rat has already moved):
        1. predict()               — propagate belief through transition matrix
        2. update()                — reweight by (noise_type, noisy_distance) observation
        3. observe_opponent_search() — optional hard update if opponent searched

    When the rat is caught, call reset_after_catch() to reinitialize belief to
    the stationary distribution of T (approximated by T^1000, computed once).
    """

    def __init__(self, T: np.ndarray):
        """
        Parameters
        ----------
        T : np.ndarray, shape (64, 64)
            Transition matrix where T[i, j] = P(rat moves from cell i to cell j).
            Row i sums to 1.
        """
        self._T = np.array(T, dtype=np.float64)  # keep our own float64 copy
        self._stationary = self._compute_stationary()

        # Start with stationary distribution — rat has already taken 1000 steps
        self._belief = self._stationary.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_stationary(self) -> np.ndarray:
        """
        Approximate the stationary distribution by raising T to a large power.
        After 1000 steps the rat's distribution has essentially converged.
        We use repeated squaring for efficiency.
        """
        # 2^10 = 1024 steps — enough for convergence on a 64-state chain
        T_power = self._T.copy()
        for _ in range(10):
            T_power = T_power @ T_power

        # Any row will do; use row 0 (all rows converge to stationary dist)
        stationary = T_power[0].copy()
        stationary /= stationary.sum()  # renormalize for floating point safety
        return stationary

    def _build_distance_likelihood(
        self, reported_dist: int, worker_pos: tuple[int, int]
    ) -> np.ndarray:
        """
        For each cell i, compute P(reported_dist | rat is at cell i, worker at worker_pos).

        The measurement model says:
            reported = actual + offset,  offset ~ {-1:0.12, 0:0.70, +1:0.12, +2:0.06}
            reported is capped at 0 from below.

        So for a given cell i with actual distance d_actual:
            P(reported=r | actual=d_actual) = sum of _DIST_ERROR_PROBS[offset]
                                              for all offsets where actual + offset == r,
                                              PLUS any probability mass folded into 0
                                              when actual + offset < 0.

        Parameters
        ----------
        reported_dist : int
            The noisy distance estimate received from rat.sample().
        worker_pos : tuple[int, int]
            (x, y) of YOUR worker this turn.

        Returns
        -------
        np.ndarray, shape (64,)
            Likelihood of the reported distance for each cell.
        """
        wx, wy = worker_pos
        likelihoods = np.zeros(64, dtype=np.float64)

        for i in range(64):
            rx, ry = _index_to_cell(i)
            d_actual = _manhattan(wx, wy, rx, ry)

            p = 0.0
            for offset, prob in _DIST_ERROR_PROBS.items():
                raw_reported = d_actual + offset
                # Cap at 0 — engine never reports negative distances
                effective_reported = max(0, raw_reported)
                if effective_reported == reported_dist:
                    p += prob

            likelihoods[i] = p

        return likelihoods

    def _build_noise_likelihood(
        self, noise_type: Noise, cell_types: list
    ) -> np.ndarray:
        """
        For each cell i, compute P(noise_type | rat is at cell i).

        Uses the floor type at cell i to look up the emission probability.
        Note: the rat CAN be under blocked squares — Blocked has its own row
        in the emission table and must not be skipped.

        Parameters
        ----------
        noise_type : Noise
            The Noise enum value (SQUEAK=0, SCRATCH=1, SQUEAL=2) from rat.sample().
        cell_types : list of length 64
            Snapshot of board cell types, cell_types[i] = board.get_cell(index_to_cell(i)).
            Pre-computed by update() to avoid 64 repeated board calls.

        Returns
        -------
        np.ndarray, shape (64,)
            Likelihood of the noise type for each cell.
        """
        noise_col = noise_type.value  # Noise is an IntEnum: SQUEAK=0, SCRATCH=1, SQUEAL=2
        likelihoods = np.zeros(64, dtype=np.float64)
        for i in range(64):
            emission_row = _NOISE_EMISSION.get(cell_types[i], _NOISE_EMISSION[Cell.SPACE])
            likelihoods[i] = emission_row[noise_col]
        return likelihoods

    def _normalize(self) -> None:
        """Renormalize belief to sum to 1. If all zero (shouldn't happen), reset to stationary."""
        total = self._belief.sum()
        if total < 1e-12:
            # Numerical collapse — something went wrong. Reset rather than divide by zero.
            self._belief = self._stationary.copy()
        else:
            self._belief /= total

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """
        Prediction step: propagate belief through the transition matrix.

        Call this FIRST each turn, because the rat has already moved before
        you receive the observation.

            belief_new[j] = sum_i belief[i] * T[i, j]
                          = belief @ T  (row-vector convention)
        """
        self._belief = self._belief @ self._T

    def update(
        self,
        noise_type: Noise,
        reported_dist: int,
        worker_pos: tuple[int, int],
        board,
    ) -> None:
        """
        Update step: reweight belief by observation likelihood.

        Parameters
        ----------
        noise_type : Noise
            The Noise enum value from rat.sample() — Noise.SQUEAK, SCRATCH, or SQUEAL.
        reported_dist : int
            The noisy manhattan distance estimate from rat.sample().
        worker_pos : tuple[int, int]
            YOUR worker's (x, y) position this turn.
        board : Board
            Current board (used to snapshot cell types for noise likelihood).
        """
        # Snapshot all 64 cell types once rather than calling board.get_cell 64 times
        # inside _build_noise_likelihood.
        cell_types = [board.get_cell(_index_to_cell(i)) for i in range(64)]

        noise_lk = self._build_noise_likelihood(noise_type, cell_types)
        dist_lk = self._build_distance_likelihood(reported_dist, worker_pos)

        # Both observations are conditionally independent given the rat's position,
        # so joint likelihood is their product.
        self._belief *= noise_lk * dist_lk
        self._normalize()

    def observe_opponent_search(self, loc: tuple[int, int], found: bool) -> None:
        """
        Hard update from opponent's search result.

        If found=True:  rat WAS at loc → set all other cells to 0, then reset
                        (new rat spawning — call reset_after_catch instead if
                         you're handling the spawn elsewhere).
        If found=False: rat was NOT at loc → zero out that cell and renormalize.

        Parameters
        ----------
        loc : tuple[int, int]
            The (x, y) cell the opponent searched.
        found : bool
            Whether the opponent found the rat there.
        """
        if loc is None:
            return  # Opponent didn't search this turn

        i = _cell_to_index(*loc)

        if found:
            # Rat was definitively at loc. A new rat will have been spawned.
            # Reset to stationary — we have no information about the new rat.
            self.reset_after_catch()
        else:
            # Rat was definitively NOT at loc.
            self._belief[i] = 0.0
            self._normalize()

    def observe_own_search(self, loc: tuple[int, int], found: bool) -> None:
        """
        Hard update from your own search result.

        Same logic as observe_opponent_search. Call this after you make a
        SEARCH move so the belief state stays consistent.

        Parameters
        ----------
        loc : tuple[int, int]
            The (x, y) cell you searched.
        found : bool
            Whether you found the rat.
        """
        if found:
            self.reset_after_catch()
        else:
            i = _cell_to_index(*loc)
            self._belief[i] = 0.0
            self._normalize()

    def reset_after_catch(self) -> None:
        """
        Reset belief after a rat catch (by either player).

        The new rat takes 1000 steps from (0,0) before the game resumes.
        After 1000 steps the distribution has converged to stationary,
        so that's our best prior for the new rat's location.
        """
        self._belief = self._stationary.copy()

    def get_belief(self) -> np.ndarray:
        """
        Returns the current belief distribution.

        Returns
        -------
        np.ndarray, shape (64,)
            Probability distribution over all 64 cells. Sums to 1.
        """
        return self._belief.copy()

    def search_ev(self, loc: tuple[int, int]) -> float:
        """
        Expected value of searching a given cell.

            EV = 4 * P(rat at loc) - 2 * (1 - P(rat at loc))
               = 6 * P(rat at loc) - 2

        A search is worth making only if EV > 0, i.e. P(rat at loc) > 1/3.

        Parameters
        ----------
        loc : tuple[int, int]
            The (x, y) cell to evaluate.

        Returns
        -------
        float
            Expected point gain from searching this cell.
        """
        p = self._belief[_cell_to_index(*loc)]
        return 6.0 * p - 2.0

    def best_search_target(self) -> tuple[tuple[int, int], float]:
        """
        Find the cell with the highest search expected value.

        Returns
        -------
        (loc, ev) : tuple[tuple[int, int], float]
            The best cell to search and its expected value.
            ev may be negative — caller should check before searching.
        """
        best_i = int(np.argmax(self._belief))
        best_loc = _index_to_cell(best_i)
        ev = self.search_ev(best_loc)
        return best_loc, ev