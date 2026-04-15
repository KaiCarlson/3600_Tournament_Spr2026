"""
tune_depth.py — Search depth tester for Design_1_Kai
=====================================================
Usage (run from the engine/ directory, same place as run_local_agents.py):

    python3 tune_depth.py

What it does
------------
1. For each depth in DEPTH_GRID, temporarily patches MAX_DEPTH in search.py.
2. Runs GAMES_PER_DEPTH games of the agent vs. itself at that depth.
3. Records per-game point totals for both players.
4. Restores the original search.py after each depth.
5. Prints a ranked summary table sorted by avg total points descending,
   then std dev ascending (higher scoring + more consistent = better).

Why these metrics and not win rate
-----------------------------------
Both sides use the same depth, so win rate is noise — first-mover advantage
and board randomness dominate. What we actually care about:

  avg_total   : average combined points per game (A + B). Higher means both
                sides are finding good carpet lines. A depth that produces
                low avg_total is playing defensively or failing to set up runs.

  |delta|     : average absolute point difference per game. Low means the
                two sides score similarly — the search is consistent and not
                wildly asymmetric. High means one side is getting lucky or
                the search has a first-mover bias at that depth.

  std_dev     : standard deviation of total points across games. Low means
                the depth produces stable results game to game. High means
                the outcome depends heavily on board randomness rather than
                search quality — a sign the depth is too shallow to overcome
                the stochastic elements.

How to extend
-------------
- Add or remove values from DEPTH_GRID.
- Increase GAMES_PER_DEPTH for more statistical power (slower).
- The patch/restore mechanism writes search.py to disk because play_game()
  forks child processes that import the agent fresh — in-memory patches are
  invisible to them.
"""

import math
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

# Number of games per depth. 10 is fast; 20-30 is more reliable.
GAMES_PER_DEPTH = 20

# Agent name (folder under 3600-agents/ that contains agent.py).
AGENT_NAME = "Design_2_Kai"

# Depths to test. Add or remove as needed.
DEPTH_GRID = [3, 4]

# ---------------------------------------------------------------------------
# Internals — no need to edit below this line
# ---------------------------------------------------------------------------

@dataclass
class DepthResult:
    depth: int
    games: int = 0
    errors: int = 0
    total_a_pts: float = 0.0
    total_b_pts: float = 0.0
    total_score_delta: float = 0.0
    # For std dev: track sum of squares of total_pts per game
    _total_pts_list: list = field(default_factory=list)

    @property
    def avg_a_pts(self) -> float:
        return self.total_a_pts / self.games if self.games else 0.0

    @property
    def avg_b_pts(self) -> float:
        return self.total_b_pts / self.games if self.games else 0.0

    @property
    def avg_total_pts(self) -> float:
        return (self.total_a_pts + self.total_b_pts) / self.games if self.games else 0.0

    @property
    def avg_score_delta(self) -> float:
        return self.total_score_delta / self.games if self.games else 0.0

    @property
    def abs_avg_delta(self) -> float:
        return abs(self.avg_score_delta)

    @property
    def std_dev_total(self) -> float:
        """Sample std dev of (A+B) total points per game."""
        if self.games < 2:
            return 0.0
        mean = self.avg_total_pts
        variance = sum((x - mean) ** 2 for x in self._total_pts_list) / (self.games - 1)
        return math.sqrt(variance)

    def label(self) -> str:
        return f"depth={self.depth}"


def _find_engine_dir() -> pathlib.Path:
    """Find the engine/ directory (where run_local_agents.py lives)."""
    here = pathlib.Path(__file__).parent.resolve()
    if (here / "run_local_agents.py").exists():
        return here
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


def _patch_depth(agent_dir: pathlib.Path, depth: int) -> str:
    """
    Overwrite MAX_DEPTH in search.py with the given value.
    Returns the original file content so we can restore it.

    Matches lines like:   MAX_DEPTH = 3
    """
    search_path = agent_dir / "search.py"
    original = search_path.read_text()

    patched = re.sub(
        r"^(MAX_DEPTH\s*=\s*)(\d+)",
        rf"\g<1>{depth}",
        original,
        flags=re.MULTILINE,
    )

    if patched == original:
        raise ValueError(
            f"Could not find MAX_DEPTH constant in {search_path}. "
            "Make sure search.py has a line like: MAX_DEPTH = 3"
        )

    search_path.write_text(patched)
    return original


def _restore_search(agent_dir: pathlib.Path, original_content: str):
    search_path = agent_dir / "search.py"
    search_path.write_text(original_content)


def _run_one_game(engine_dir: pathlib.Path) -> Optional[dict]:
    """
    Run a single game of AGENT_NAME vs itself.
    Returns a dict with result info, or None on error.
    """
    engine_str = str(engine_dir)
    agents_root = str(engine_dir.parent / "3600-agents")
    for p in [engine_str, agents_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from gameplay import play_game

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

    if board.is_player_a_turn:
        a_pts = board.player_worker.get_points()
        b_pts = board.opponent_worker.get_points()
    else:
        b_pts = board.player_worker.get_points()
        a_pts = board.opponent_worker.get_points()

    return {
        "winner": winner,
        "a_pts": a_pts,
        "b_pts": b_pts,
        "turns": board.turn_count,
        "reason": board.win_reason,
    }


def run_depth(depth: int, engine_dir: pathlib.Path,
              agent_dir: pathlib.Path, n_games: int) -> DepthResult:
    result = DepthResult(depth=depth)

    print(f"\n[Depth {depth}]")
    print(f"  Running {n_games} games...")

    original = _patch_depth(agent_dir, depth)
    try:
        for g in range(n_games):
            game_result = _run_one_game(engine_dir)
            if game_result is None:
                result.errors += 1
                continue

            result.games += 1
            a_pts = game_result["a_pts"]
            b_pts = game_result["b_pts"]
            total = a_pts + b_pts

            result.total_a_pts += a_pts
            result.total_b_pts += b_pts
            result.total_score_delta += (a_pts - b_pts)
            result._total_pts_list.append(total)

            from game.enums import ResultArbiter
            w = game_result["winner"]
            winner_str = (
                "A wins" if w == ResultArbiter.PLAYER_A else
                "B wins" if w == ResultArbiter.PLAYER_B else
                "TIE   "
            )
            print(f"  Game {g+1:2d}: A={a_pts:+d} B={b_pts:+d} total={total:+d} "
                  f"{winner_str} ({game_result['reason'].name})")
    finally:
        _restore_search(agent_dir, original)

    print(f"  => avg_total={result.avg_total_pts:.1f}  "
          f"std_dev={result.std_dev_total:.1f}  "
          f"|delta|={result.abs_avg_delta:.1f}  "
          f"avg_A={result.avg_a_pts:.1f}  avg_B={result.avg_b_pts:.1f}  "
          f"errors={result.errors}")
    return result


def print_summary(results: list[DepthResult]):
    print("\n" + "=" * 90)
    print("SUMMARY — sorted by avg_total descending, then std_dev ascending")
    print("Goal: high avg_total = both sides scoring well  |  low std_dev = consistent play")
    print("=" * 90)

    sorted_results = sorted(results, key=lambda r: (-r.avg_total_pts, r.std_dev_total))

    header = (f"{'#':>3}  {'depth':>5}  {'avgTotal':>9}  {'stdDev':>7}  "
              f"{'|ΔA-B|':>7}  {'avgA':>6}  {'avgB':>6}  {'games':>5}  {'errors':>6}")
    print(header)
    print("-" * 90)
    for rank, r in enumerate(sorted_results, 1):
        print(f"{rank:>3}  {r.depth:>5}  {r.avg_total_pts:>9.1f}  {r.std_dev_total:>7.1f}  "
              f"{r.abs_avg_delta:>7.1f}  "
              f"{r.avg_a_pts:>6.1f}  {r.avg_b_pts:>6.1f}  "
              f"{r.games:>5}  {r.errors:>6}")

    print(f"\nRecommended depth: {sorted_results[0].depth}")
    print(
        "Note: if depth 3 and 4 score similarly but depth 4 has higher std_dev,\n"
        "the extra lookahead is adding noise, not value — stick with depth 3.\n"
        "If depth 4 scores clearly higher AND has lower std_dev, raise MAX_DEPTH."
    )


def main():
    multiprocessing.set_start_method("spawn", force=True)

    engine_dir = _find_engine_dir()
    agent_dir = _agent_dir(engine_dir)

    print(f"Engine dir  : {engine_dir}")
    print(f"Agent dir   : {agent_dir}")
    print(f"Games/depth : {GAMES_PER_DEPTH}")
    print(f"Depths      : {DEPTH_GRID}")
    print(f"Total games : {len(DEPTH_GRID) * GAMES_PER_DEPTH}")

    t0 = time.perf_counter()
    results = []

    for depth in DEPTH_GRID:
        r = run_depth(depth, engine_dir, agent_dir, GAMES_PER_DEPTH)
        results.append(r)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print_summary(results)


if __name__ == "__main__":
    main()