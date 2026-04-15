"""
tune_heuristic.py — Heuristic weight tuner for Design_1_Kai
============================================================
Usage (run from the engine/ directory, same place as run_local_agents.py):

    python3 tune_heuristic.py

What it does
------------
1. Defines a grid of weight combinations to test.
2. For each combination, temporarily writes a patched heuristic.py into
   the agent folder, runs N games of the agent vs. itself, and records:
     - win rate for player A
     - average score delta (A_pts - B_pts)
3. Restores the original heuristic.py after each trial.
4. Prints a ranked summary table at the end.

Why patch heuristic.py on disk instead of monkey-patching in memory
--------------------------------------------------------------------
play_game() forks separate processes with multiprocessing (spawn mode).
Those child processes import the agent fresh from disk, so in-memory
patches are invisible to them. Writing the file is the only reliable way
to make the child processes see the new weights.

How to extend
-------------
- Add more weight combinations to WEIGHT_GRID.
- Increase GAMES_PER_CONFIG for more statistical power (slower).
- Set PARALLEL=True to run configs concurrently (uses more CPU).
  With PARALLEL=True, DO NOT also parallelise inside play_game — the
  agent processes already use multiprocessing internally.
"""

import copy
import os
import pathlib
import re
import sys
import time
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Number of games per weight configuration. 10 is fast; 30+ is more reliable.
GAMES_PER_CONFIG = 30

# Agent name (folder under 3600-agents/ that contains agent.py).
AGENT_NAME = "Design_1_Kai"

# Run configs sequentially (False) or in parallel (True).
# Parallel is faster but noisier; set False if you hit import conflicts.
PARALLEL = False

# ---------------------------------------------------------------------------
# Weight grid — list of dicts, each dict is one configuration to test.
# Keys must match the constant names at the top of heuristic.py exactly.
# ---------------------------------------------------------------------------

WEIGHT_GRID = [
    # New winner — exact rerun for confirmation
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.5, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.78, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.6),

    # Roll nudge up — does 2.5 plateau or keep climbing?
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.8, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.78, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.6),

    # Roll nudge down slightly — is 2.5 a peak or a floor?
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.2, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.78, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.6),

    # Decay tighter — winner was 0.78, previous winner was 0.75. Is 0.76 better?
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.5, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.76, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.6),

    # Decay looser by a hair — 0.78 beat 0.80 badly, but is 0.79 still ok?
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.5, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.79, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.6),

    # Opp nudge down — 0.7 hurt badly, 0.6 won, 0.5 was decent. Is 0.55 a sweet spot?
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.5, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.78, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.55),

    # Opp nudge up cautiously — 0.65, before the 0.7 cliff
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.5, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.78, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.65),

    # Primed even lower — 0.4 has consistently beaten everything higher. Try 0.2.
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.5, W_PRIMED_FUTURE=0.2,
         DISTANCE_DECAY=0.78, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.6),

    # Rat EV bump — hasn't been tested near the winner. Does 1.2 help?
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.5, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.78, W_RAT_EV=1.2, W_OPPONENT_ROLLABLE=0.6),

    # Best guess combo — roll up, decay tighter, opp slightly trimmed
    dict(W_SCORE_DELTA=10.0, W_ROLLABLE_POTENTIAL=2.8, W_PRIMED_FUTURE=0.4,
         DISTANCE_DECAY=0.76, W_RAT_EV=1.0, W_OPPONENT_ROLLABLE=0.6),
]

# ---------------------------------------------------------------------------
# Internals — no need to edit below this line
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    weights: dict
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    games: int = 0
    total_score_delta: float = 0.0
    total_a_pts: float = 0.0
    total_b_pts: float = 0.0
    errors: int = 0

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        # Both A and B use the same weights, so winning as either side counts.
        # Since the board is symmetric, we track A's win rate.
        return self.wins_a / self.games

    @property
    def avg_score_delta(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_score_delta / self.games

    @property
    def avg_a_pts(self) -> float:
        return self.total_a_pts / self.games if self.games else 0.0

    @property
    def avg_b_pts(self) -> float:
        return self.total_b_pts / self.games if self.games else 0.0

    @property
    def avg_total_pts(self) -> float:
        """Average combined points per game (A + B). Higher = both bots scoring well."""
        return (self.total_a_pts + self.total_b_pts) / self.games if self.games else 0.0

    @property
    def abs_avg_delta(self) -> float:
        """Average absolute score difference per game. Lower = more consistent/even games."""
        return abs(self.avg_score_delta)

    def label(self) -> str:
        w = self.weights
        return (f"roll={w['W_ROLLABLE_POTENTIAL']:.1f} "
                f"primed={w['W_PRIMED_FUTURE']:.1f} "
                f"decay={w['DISTANCE_DECAY']:.2f} "
                f"rat={w['W_RAT_EV']:.1f} "
                f"opp={w['W_OPPONENT_ROLLABLE']:.1f}")


def _find_engine_dir() -> pathlib.Path:
    """Find the engine/ directory (where run_local_agents.py lives)."""
    here = pathlib.Path(__file__).parent.resolve()
    # If run from engine/, we're already there
    if (here / "run_local_agents.py").exists():
        return here
    # If run from the repo root
    engine = here / "engine"
    if (engine / "run_local_agents.py").exists():
        return engine
    raise FileNotFoundError(
        "Cannot find run_local_agents.py. "
        "Run this script from the engine/ directory."
    )


def _agent_dir(engine_dir: pathlib.Path) -> pathlib.Path:
    agents_root = engine_dir.parent / "3600-agents"
    agent = agents_root / AGENT_NAME
    if not agent.exists():
        raise FileNotFoundError(f"Agent directory not found: {agent}")
    return agent


def _patch_heuristic(agent_dir: pathlib.Path, weights: dict) -> str:
    """
    Overwrite the weight constants at the top of heuristic.py with new values.
    Returns the original file content so we can restore it.
    """
    heuristic_path = agent_dir / "heuristic.py"
    original = heuristic_path.read_text()

    patched = original
    for name, value in weights.items():
        # Match lines like:   W_ROLLABLE_POTENTIAL = 2.0
        pattern = rf"^({re.escape(name)}\s*=\s*)([\d.]+)"
        replacement = rf"\g<1>{value}"
        patched = re.sub(pattern, replacement, patched, flags=re.MULTILINE)

    heuristic_path.write_text(patched)
    return original


def _restore_heuristic(agent_dir: pathlib.Path, original_content: str):
    heuristic_path = agent_dir / "heuristic.py"
    heuristic_path.write_text(original_content)


def _run_one_game(engine_dir: pathlib.Path) -> Optional[dict]:
    """
    Run a single game of AGENT_NAME vs itself.
    Returns a dict with result info, or None on error.
    """
    # We must add engine_dir to sys.path so gameplay.py imports work
    engine_str = str(engine_dir)
    agents_root = str(engine_dir.parent / "3600-agents")
    for p in [engine_str, agents_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from gameplay import play_game  # imported here to pick up patched heuristic

    try:
        board, _, spawn_a, spawn_b, msg_a, msg_b = play_game(
            agents_root,
            agents_root,
            AGENT_NAME,
            AGENT_NAME,
            display_game=False,
            delay=0.0,
            clear_screen=False,
            record=False,
            limit_resources=False,
        )
    except Exception as e:
        print(f"  [ERROR] play_game raised: {e}")
        return None

    from game.enums import ResultArbiter
    winner = board.get_winner()

    # Determine A and B final points
    # After game ends, is_player_a_turn tells us whose turn it technically is,
    # but points are tracked absolutely on each worker.
    if board.is_player_a_turn:
        a_pts = board.player_worker.get_points()
        b_pts = board.opponent_worker.get_points()
    else:
        b_pts = board.player_worker.get_points()
        a_pts = board.opponent_worker.get_points()

    return {
        "winner": winner,        # ResultArbiter enum
        "a_pts": a_pts,
        "b_pts": b_pts,
        "turns": board.turn_count,
        "reason": board.win_reason,
    }


def run_config(config_index: int, weights: dict, engine_dir: pathlib.Path,
               agent_dir: pathlib.Path, n_games: int) -> ConfigResult:
    result = ConfigResult(weights=weights)

    print(f"\n[Config {config_index}] {result.label()}")
    print(f"  Running {n_games} games...")

    original = _patch_heuristic(agent_dir, weights)
    try:
        for g in range(n_games):
            game_result = _run_one_game(engine_dir)
            if game_result is None:
                result.errors += 1
                continue

            result.games += 1
            a_pts = game_result["a_pts"]
            b_pts = game_result["b_pts"]
            result.total_a_pts += a_pts
            result.total_b_pts += b_pts
            result.total_score_delta += (a_pts - b_pts)

            from game.enums import ResultArbiter
            w = game_result["winner"]
            if w == ResultArbiter.PLAYER_A:
                result.wins_a += 1
            elif w == ResultArbiter.PLAYER_B:
                result.wins_b += 1
            else:
                result.ties += 1

            print(f"  Game {g+1:2d}: A={a_pts:+d} B={b_pts:+d} "
                  f"{'A wins' if w==ResultArbiter.PLAYER_A else 'B wins' if w==ResultArbiter.PLAYER_B else 'TIE':6s} "
                  f"({game_result['reason'].name})")
    finally:
        _restore_heuristic(agent_dir, original)

    print(f"  => avg_total={result.avg_total_pts:.1f}  "
          f"|delta|={result.abs_avg_delta:.1f}  "
          f"avg_A={result.avg_a_pts:.1f}  avg_B={result.avg_b_pts:.1f}  "
          f"errors={result.errors}")
    return result


def print_summary(results: list[ConfigResult]):
    print("\n" + "="*100)
    print("SUMMARY — sorted by avg total points (A+B) descending, then by |ΔA-B| ascending")
    print("Goal: high avg_total = both bots scoring well; low |delta| = consistent, not lopsided")
    print("="*100)
    # Primary sort: avg_total_pts descending (higher combined scoring = better strategy)
    # Secondary sort: abs_avg_delta ascending (lower spread = more consistent)
    sorted_results = sorted(results, key=lambda r: (-r.avg_total_pts, r.abs_avg_delta))

    header = (f"{'#':>3}  {'avgTotal':>9}  {'|ΔA-B|':>7}  {'ΔA-B':>6}  {'avgA':>6}  {'avgB':>6}  "
              f"{'W':>3}{'T':>3}{'L':>3}  weights")
    print(header)
    print("-" * 100)
    for rank, r in enumerate(sorted_results, 1):
        print(f"{rank:>3}  {r.avg_total_pts:>9.1f}  {r.abs_avg_delta:>7.1f}  "
              f"{r.avg_score_delta:>+6.1f}  "
              f"{r.avg_a_pts:>6.1f}  {r.avg_b_pts:>6.1f}  "
              f"{r.wins_a:>3}{r.ties:>3}{r.wins_b:>3}  {r.label()}")

    print("\nBest config weights to paste into heuristic.py:")
    best = sorted_results[0]
    for k, v in best.weights.items():
        print(f"  {k} = {v}")


def main():
    multiprocessing.set_start_method("spawn", force=True)

    engine_dir = _find_engine_dir()
    agent_dir = _agent_dir(engine_dir)

    print(f"Engine dir : {engine_dir}")
    print(f"Agent dir  : {agent_dir}")
    print(f"Games/config: {GAMES_PER_CONFIG}")
    print(f"Configs    : {len(WEIGHT_GRID)}")
    print(f"Total games: {len(WEIGHT_GRID) * GAMES_PER_CONFIG}")

    t0 = time.perf_counter()
    results = []

    for i, weights in enumerate(WEIGHT_GRID):
        r = run_config(i, weights, engine_dir, agent_dir, GAMES_PER_CONFIG)
        results.append(r)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print_summary(results)


if __name__ == "__main__":
    main()