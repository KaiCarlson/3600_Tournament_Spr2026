import numpy as np
from game.enums import Cell, Direction, BOARD_SIZE, CARPET_POINTS_TABLE, loc_after_direction

# ---------------------------------------------------------------------------
# Tunable weights — adjust these empirically by running games.
# Every magic number in this file should live here, not buried in expressions.
# ---------------------------------------------------------------------------

# How much to weight the raw score delta (your points minus opponent's).
# This should always be the dominant term — real points beat potential.
W_SCORE_DELTA = 10.0

# How much to weight each unit of immediately rollable carpet potential.
# "Immediately rollable" means a contiguous primed run you could carpet right now.
# Weighted by the points that roll would actually score.
W_ROLLABLE_POTENTIAL = 2.0

# How much to weight primed cells that are NOT yet rollable (future setup value).
# Discounted by manhattan distance from your worker, since far cells may never pay off.
W_PRIMED_FUTURE = 0.4

# Distance decay base for future primed cell value.
# Value of a primed cell k steps away = W_PRIMED_FUTURE * (DISTANCE_DECAY ^ k)
# Set < 1.0 so distant cells matter less. Closer to 0 = more aggressive discounting.
DISTANCE_DECAY = 0.75

# How much to weight the best available rat search EV.
# Rat EV is already in point units (6p - 2), so this should stay near 1.0.
# Lowering it makes the bot less likely to chase the rat.
W_RAT_EV = 1.0

# How much to penalize the opponent having rollable potential.
# Setting this > 0 makes the bot account for the opponent's threats.
# Start low — overweighting this can cause overly defensive play.
W_OPPONENT_ROLLABLE = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rollable_length(board, direction: Direction) -> int:
    """
    From the current player's position, walk in `direction` and count how many
    contiguous primed cells exist that could be carpeted in a single roll.

    Returns the maximum roll length achievable in that direction right now.
    Returns 0 if the first cell in that direction is not primed/carpetable.
    """
    pos = board.player_worker.get_location()
    length = 0
    current = pos
    for _ in range(1, BOARD_SIZE):
        current = loc_after_direction(current, direction)
        if not board.is_cell_carpetable(current):
            break
        length += 1
    return length


def _rollable_potential_score(board) -> float:
    """
    For each direction, compute the points the player would score if they
    carpet-rolled the maximum available run right now.

    This is the most immediate form of carpet value: runs the player can
    execute on THIS turn or the next few turns.

    Returns the sum of achievable carpet scores across all four directions.
    Note: roll of length 1 scores -1 (a bad move). We include it here because
    the heuristic should accurately reflect the board state — the caller
    (search.py) will simply not pick it as the best move. Alternatively we
    could floor at 0, but that would artificially inflate positions with only
    length-1 runs.
    """
    total = 0.0
    for direction in Direction:
        length = _rollable_length(board, direction)
        if length >= 1:
            total += CARPET_POINTS_TABLE[length]
    return total


def _future_primed_value(board) -> float:
    """
    Estimate the value of primed cells that are NOT immediately rollable.
    These represent setup work already done that could pay off in future turns.

    Each primed cell contributes W_PRIMED_FUTURE * DISTANCE_DECAY^distance,
    where distance is the Manhattan distance from the player's current position.

    We use the bitboard directly for speed rather than calling get_cell 64 times.
    """
    pos = board.player_worker.get_location()
    px, py = pos

    primed_mask = board._primed_mask
    if primed_mask == 0:
        return 0.0

    total = 0.0
    mask = primed_mask
    while mask:
        # Isolate lowest set bit
        lsb = mask & (-mask)
        mask ^= lsb

        # Convert bit index to (x, y)
        bit_index = lsb.bit_length() - 1
        cx = bit_index % BOARD_SIZE
        cy = bit_index // BOARD_SIZE

        dist = abs(cx - px) + abs(cy - py)
        total += DISTANCE_DECAY ** dist

    return total


def _opponent_rollable_potential(board) -> float:
    """
    Estimate the opponent's immediately rollable carpet potential.
    We temporarily reverse perspective to reuse _rollable_potential_score,
    then reverse back. This is safe because we don't mutate any game state —
    reverse_perspective only swaps worker references.
    """
    board.reverse_perspective()
    opp_score = _rollable_potential_score(board)
    board.reverse_perspective()
    return opp_score


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(board, belief_array: np.ndarray) -> float:
    """
    Evaluate a board state from the perspective of `board.player_worker`.

    Always call this AFTER forecast_move() and reverse_perspective() so that
    `player_worker` refers to the player whose position you are scoring.

    Parameters
    ----------
    board : Board
        The board state to evaluate. player_worker is the player being scored.
    belief_array : np.ndarray, shape (64,)
        Current rat belief distribution (from RatBelief.get_belief()).
        Used to compute rat search EV contribution.

    Returns
    -------
    float
        Scalar heuristic score. Higher = better for player_worker.
    """
    # --- Component 1: Raw score delta ---
    # This is the only ground truth. Everything else is estimation.
    score_delta = (
        board.player_worker.get_points() - board.opponent_worker.get_points()
    )

    # --- Component 2: Immediately rollable carpet potential ---
    # Points achievable if the player carpet-rolls right now in each direction.
    rollable = _rollable_potential_score(board)

    # --- Component 3: Future primed cell value ---
    # Discounted by distance — far primed cells may never pay off.
    future_primed = _future_primed_value(board)

    # --- Component 4: Opponent rollable potential (threat) ---
    # Penalize positions where the opponent has strong immediate carpet options.
    opp_rollable = _opponent_rollable_potential(board)

    # --- Component 5: Rat EV ---
    # Best expected value from a single search, contributed as a positional bonus.
    # We don't issue the SEARCH move here — that's search.py's job.
    # We just reward positions where the belief is spiked enough to matter.
    best_idx = int(np.argmax(belief_array))
    best_p = float(belief_array[best_idx])
    rat_ev = 6.0 * best_p - 2.0  # EV formula: 4p - 2(1-p) = 6p - 2
    rat_ev_contrib = max(0.0, rat_ev)  # Only positive EV is a bonus

    # --- Weighted sum ---
    return (
        W_SCORE_DELTA       * score_delta
        + W_ROLLABLE_POTENTIAL * rollable
        + W_PRIMED_FUTURE      * future_primed
        - W_OPPONENT_ROLLABLE  * opp_rollable
        + W_RAT_EV             * rat_ev_contrib
    )


def turns_left_discount(turns_left: int) -> float:
    """
    A multiplier (0.0–1.0) that scales down potential-based terms
    when few turns remain. When turns_left is high, potential matters.
    When turns_left is 1 or 2, only immediate scoring matters.

    This is provided as a utility for search.py to optionally apply
    to the potential components when evaluating late-game states.
    Currently not used inside evaluate() — keeping them separate gives
    you the option to tune urgency scaling independently.

    Parameters
    ----------
    turns_left : int
        The player's remaining turn count.

    Returns
    -------
    float
        Discount multiplier between 0.0 and 1.0.
    """
    from game.enums import MAX_TURNS_PER_PLAYER
    # Linear decay: full weight at max turns, zero at 0 turns.
    return max(0.0, turns_left / MAX_TURNS_PER_PLAYER)