import numpy as np
from game.move import Move
from game.enums import MoveType
from .heuristic import evaluate, W_SCORE_DELTA
from .rat_belief import RatBelief

# ---------------------------------------------------------------------------
# Tunable search parameters
# ---------------------------------------------------------------------------

# Hard time reserve — abort search and return best-so-far if time drops below this.
# Must be large enough to cover the cost of one final evaluate() call plus
# any overhead in agent.py after best_move() returns.
TIME_RESERVE = 0.15

# Maximum search depth for iterative deepening.
# Depth 1 = greedy (Tier 1 behavior).
# Depth 2-3 = practical range given ~6s/move budget and branching factor ~20.
# Iterative deepening will stop early if time runs out, so this is a ceiling.
MAX_DEPTH = 3

# Minimum rat search EV (in raw points) to consider a search move at all.
# EV = 6p - 2, so threshold 0.0 means search whenever p > 1/3.
RAT_SEARCH_EV_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Public interface — unchanged from Tier 1, agent.py needs no edits
# ---------------------------------------------------------------------------

def best_move(board, belief: RatBelief, time_left_func) -> Move:
    """
    Select the best move for the current player using iterative deepening
    negamax with alpha-beta pruning.

    The rat search decision is handled outside the tree: we compare the best
    tree move against the rat search EV at the end, in heuristic-scaled units.

    Parameters
    ----------
    board : Board
        Current board state. Not mutated.
    belief : RatBelief
        Current rat belief tracker. Read-only here.
    time_left_func : callable
        Returns remaining turn time in seconds when called.

    Returns
    -------
    Move
        The selected move.
    """
    belief_array = belief.get_belief()

    # --- Rat search candidate ---
    search_loc, search_ev = belief.best_search_target()

    # --- Iterative deepening ---
    # Start at depth 1, deepen until time runs out or MAX_DEPTH is reached.
    # best_move_so_far is always a valid fallback from the last completed depth.
    best_move_so_far = None
    best_score_so_far = float('-inf')

    for depth in range(1, MAX_DEPTH + 1):
        if time_left_func() < TIME_RESERVE:
            break

        move, score = _negamax_root(board, belief_array, depth, time_left_func)

        if move is not None:
            best_move_so_far = move
            best_score_so_far = score

        if move is None or time_left_func() < TIME_RESERVE:
            break

    # --- Compare tree best against rat search ---
    # search_ev is in raw point units (6p - 2).
    # best_score_so_far is in heuristic units — dividing by W_SCORE_DELTA converts
    # it back to approximate raw points so both sides are on the same scale.
    # The old code compared search_ev * W_SCORE_DELTA against best_score_so_far
    # directly, which meant search almost never triggered because best_score_so_far
    # is a large absolute value (includes accumulated score delta * 10).
    if search_ev > RAT_SEARCH_EV_THRESHOLD:
        tree_marginal_pts = best_score_so_far / W_SCORE_DELTA
        if best_move_so_far is None or search_ev > tree_marginal_pts:
            return Move.search(search_loc)

    if best_move_so_far is not None:
        return best_move_so_far

    # Absolute fallback — should never reach here in a valid game state.
    return Move.search(search_loc)


# ---------------------------------------------------------------------------
# Internal search
# ---------------------------------------------------------------------------

def _negamax_root(
    board, belief_array: np.ndarray, depth: int, time_left_func
) -> tuple[Move | None, float]:
    """
    Negamax root: iterate over all moves and return (best_move, best_score).

    Separated from _negamax so we can track which root move leads to the
    best score. Interior nodes only need scores, not move objects.

    Returns (None, -inf) if time ran out before any move was fully evaluated,
    signaling to the caller that this depth is incomplete.

    Perspective note:
        At the root, board.player_worker = us. Scores returned by _negamax
        are from the perspective of the player to move at each node (negamax
        convention). At the root that's us, so scores are directly our score —
        higher is better.
    """
    moves = board.get_valid_moves(exclude_search=True)

    best_move_found = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    for move in moves:
        if time_left_func() < TIME_RESERVE:
            # Incomplete depth — signal with None.
            return None, float('-inf')

        next_board = board.forecast_move(move, check_ok=False)
        if next_board is None:
            continue

        # After forecast_move(), player_worker is still us (the one who just moved).
        # Reverse perspective so the recursive call sees the board from the
        # opponent's point of view (opponent becomes player_worker).
        next_board.reverse_perspective()

        # Recurse. The opponent maximizes their score = minimizes ours.
        # Negate their returned score to get our score from this position.
        # Swap and negate alpha/beta per the negamax convention.
        score = -_negamax(next_board, belief_array, depth - 1, -beta, -alpha, time_left_func)

        if score > best_score:
            best_score = score
            best_move_found = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

    return best_move_found, best_score


def _negamax(
    board,
    belief_array: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    time_left_func,
) -> float:
    """
    Negamax with alpha-beta pruning.

    Always returns a score from the perspective of board.player_worker —
    the player whose turn it is at this node. Higher = better for that player.

    The caller negates the returned score to convert it to their own perspective.

    Alpha-beta semantics:
        alpha = best score the current player can guarantee from here onward.
        beta  = best score the opponent can guarantee (= our ceiling).
        When alpha >= beta, the opponent would never allow this line — prune.

    Perspective management:
        forecast_move() does NOT call reverse_perspective(). After it returns,
        player_worker still refers to the mover. We call reverse_perspective()
        before recursing so the child node sees the board from the next
        player's perspective. The negation on the returned score handles the
        perspective flip: our_score = -child_score.

    Parameters
    ----------
    board : Board
        Board at this node. player_worker = the player to move here.
        This is a copy and is not shared with other nodes.
    belief_array : np.ndarray, shape (64,)
        Frozen belief snapshot from the root. The HMM is only updated once
        per real turn in agent.py — we do not update it inside the tree.
    depth : int
        Remaining depth. 0 = leaf, call evaluate() and return.
    alpha, beta : float
        Current alpha-beta window, already swapped+negated by caller.
    time_left_func : callable
        Engine time budget.

    Returns
    -------
    float
        Score from the perspective of board.player_worker at this node.
    """
    # --- Emergency time check ---
    if time_left_func() < TIME_RESERVE:
        return evaluate(board, belief_array)

    # --- Leaf node ---
    if depth == 0 or board.is_game_over():
        return evaluate(board, belief_array)

    moves = board.get_valid_moves(exclude_search=True)

    if not moves:
        return evaluate(board, belief_array)

    best_score = float('-inf')

    for move in moves:
        if time_left_func() < TIME_RESERVE:
            # Return what we have. Partial is better than crashing.
            break

        next_board = board.forecast_move(move, check_ok=False)
        if next_board is None:
            continue

        next_board.reverse_perspective()
        score = -_negamax(next_board, belief_array, depth - 1, -beta, -alpha, time_left_func)

        if score > best_score:
            best_score = score

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

    # If we evaluated nothing (shouldn't happen given the moves check above),
    # fall back to static eval rather than returning -inf.
    if best_score == float('-inf'):
        return evaluate(board, belief_array)

    return best_score