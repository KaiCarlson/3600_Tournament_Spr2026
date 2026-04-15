import numpy as np
from game.move import Move
from game.enums import MoveType
from .heuristic import evaluate, W_SCORE_DELTA
from .rat_belief import RatBelief

# ---------------------------------------------------------------------------
# Tunable search parameters
# ---------------------------------------------------------------------------

# Time (in seconds) to keep in reserve before returning best-so-far.
# If time_left_func() drops below this, abort search immediately.
# Set conservatively — a timed-out move loses the game.
TIME_RESERVE = 0.1

# Minimum rat search EV to override a carpet move.
# At 0.0, we search whenever EV > 0 (p > 1/3).
# Raising this requires higher confidence before we sacrifice a carpet turn.
# Tunable: if your bot searches too aggressively and loses points, raise this.
RAT_SEARCH_EV_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def best_move(board, belief: RatBelief, time_left_func) -> Move:
    """
    Select the best move for the current player.

    Decision flow:
        1. Check if a rat search has positive EV. If so, and if it beats
           the best carpet/prime/plain move, return the search move.
        2. Otherwise, evaluate all legal non-search moves and return the
           highest-scoring one (greedy 1-ply).
        3. If time runs out mid-search, return the best move found so far.
        4. If no moves are available (shouldn't happen), return a plain move
           in the first valid direction as a fallback.

    Parameters
    ----------
    board : Board
        Current board state. Not mutated — forecast_move() works on copies.
    belief : RatBelief
        Current rat belief tracker. Used to read belief array and compute
        search EV. Not mutated here.
    time_left_func : callable
        Returns remaining time in seconds when called. Provided by the engine.

    Returns
    -------
    Move
        The selected move.
    """
    belief_array = belief.get_belief()

    # --- Step 1: Rat search decision ---
    # Evaluate this before the move loop. If search EV clears the threshold,
    # we need to decide whether it's worth more than the best carpet move.
    # We get the best search target first, then compare against carpet options.
    search_loc, search_ev = belief.best_search_target()
    should_search = search_ev > RAT_SEARCH_EV_THRESHOLD

    # --- Step 2: Evaluate all non-search moves greedily ---
    best_nonsearch_move, best_nonsearch_score = _greedy_best_move(
        board, belief_array, time_left_func
    )

    # --- Step 3: Compare search vs best non-search move ---
    # Only issue a search if its EV beats whatever we'd score from moving.
    # Note: search_ev is in raw point units. best_nonsearch_score is heuristic
    # units (weighted). To compare them directly we'd need to normalize —
    # instead, we use a simpler rule: search only if EV > threshold AND
    # the best non-search move scores below a carpet-equivalent in heuristic.
    # For Tier 1 this is a reasonable approximation. Tier 2 will handle this
    # more cleanly by including search as a node in the expectiminimax tree.
    if should_search:
        # Convert search EV to approximate heuristic scale for comparison.
        # W_SCORE_DELTA in heuristic.py is 10.0, so 1 raw point ≈ 10 heuristic units.
        search_heuristic_equiv = search_ev * W_SCORE_DELTA

        if best_nonsearch_move is None or search_heuristic_equiv > best_nonsearch_score:
            return Move.search(search_loc)

    # --- Step 4: Return best non-search move, with fallback ---
    if best_nonsearch_move is not None:
        return best_nonsearch_move

    # Fallback: should never reach here in a valid game state, but if we do,
    # return a search on the highest-belief cell rather than crashing.
    return Move.search(search_loc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _greedy_best_move(
    board, belief_array: np.ndarray, time_left_func
) -> tuple[Move | None, float]:
    """
    Evaluate all legal non-search moves and return the best one.

    Uses 1-ply lookahead: apply each move via forecast_move(), evaluate the
    resulting board with the heuristic, return the highest-scoring move.

    Board perspective note:
        forecast_move() applies the move and calls end_turn(), which flips
        is_player_a_turn but does NOT call reverse_perspective(). So after
        forecast_move(), player_worker still refers to the player who just
        moved. evaluate() scores board.player_worker, so we are evaluating
        our own resulting position — correct for 1-ply greedy.

        When this becomes expectiminimax in Tier 2, the opponent's response
        will require reverse_perspective() before recursing. At that point,
        scores at opponent nodes will need to be negated (negamax convention)
        or handled with explicit min logic.

    Parameters
    ----------
    board : Board
        Current board state.
    belief_array : np.ndarray, shape (64,)
        Belief distribution snapshot (already fetched by best_move()).
    time_left_func : callable
        Engine time budget function.

    Returns
    -------
    (best_move, best_score) : tuple[Move | None, float]
        best_move is None only if the move list is empty (shouldn't happen).
    """
    moves = board.get_valid_moves(exclude_search=True)

    best_move_found = None
    best_score = float('-inf')

    for move in moves:
        # Abort if time is running low — return best found so far
        if time_left_func() < TIME_RESERVE:
            break

        next_board = board.forecast_move(move, check_ok=False)
        if next_board is None:
            # forecast_move returns None if the move is invalid.
            # exclude_search=True + get_valid_moves should prevent this,
            # but guard anyway.
            continue

        score = evaluate(next_board, belief_array)

        if score > best_score:
            best_score = score
            best_move_found = move

    return best_move_found, best_score