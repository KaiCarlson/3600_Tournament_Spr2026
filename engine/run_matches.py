"""
run_matches.py — Run N games between two agents and print a summary.
=====================================================================
Usage (run from the engine/ directory):

    python3 run_matches.py AgentA AgentB
    python3 run_matches.py AgentA AgentB 50

If N is omitted it defaults to NUM_GAMES below. You can also just edit
AGENT_A and AGENT_B directly and run with no arguments.
"""

import math
import pathlib
import sys
import time
import multiprocessing
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration — edit these or pass agents as command-line arguments
# ---------------------------------------------------------------------------

AGENT_A = "Design_2_Kai"
AGENT_B = "Design_2_Kai"
NUM_GAMES = 30

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _find_engine_dir() -> pathlib.Path:
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


def _run_one_game(engine_dir: pathlib.Path, agent_a: str, agent_b: str) -> Optional[dict]:
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
            agent_a,
            agent_b,
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

    # gameplay.py resolves the winner after the last reverse_perspective()
    # is NOT called (it's skipped when is_game_over()). The comment in
    # gameplay.py says "might seem reversed but this is correct" — when
    # is_player_a_turn is True, player_worker is actually B (the flip
    # happened inside apply_move/end_turn but reverse_perspective was
    # skipped). Mirror that exact logic for point extraction.
    if board.is_player_a_turn:
        # player_worker = B, opponent_worker = A
        a_pts = board.opponent_worker.get_points()
        b_pts = board.player_worker.get_points()
    else:
        # player_worker = A, opponent_worker = B
        a_pts = board.player_worker.get_points()
        b_pts = board.opponent_worker.get_points()

    return {
        "winner": winner,
        "a_pts": a_pts,
        "b_pts": b_pts,
        "msg_a": msg_a,
        "msg_b": msg_b,
        "reason": board.win_reason,
    }


def main(agent_a: str, agent_b: str, n_games: int):
    engine_dir = _find_engine_dir()

    same_agent = agent_a == agent_b
    print(f"{'Agent A':>10} : {agent_a}")
    print(f"{'Agent B':>10} : {agent_b}")
    print(f"{'Games':>10} : {n_games}")
    print()

    wins_a = wins_b = ties = errors = 0
    total_a = total_b = 0.0
    total_pts_list = []
    total_delta = 0.0

    from game.enums import ResultArbiter

    t0 = time.perf_counter()

    for g in range(n_games):
        result = _run_one_game(engine_dir, agent_a, agent_b)

        if result is None:
            errors += 1
            print(f"  Game {g+1:3d}: ERROR")
            continue

        a_pts = result["a_pts"]
        b_pts = result["b_pts"]
        total = a_pts + b_pts
        delta = a_pts - b_pts
        w = result["winner"]

        total_a += a_pts
        total_b += b_pts
        total_pts_list.append(total)
        total_delta += delta

        if w == ResultArbiter.PLAYER_A:
            wins_a += 1
            winner_str = "A wins"
        elif w == ResultArbiter.PLAYER_B:
            wins_b += 1
            winner_str = "B wins"
        else:
            ties += 1
            winner_str = "TIE   "

        msg_a = result["msg_a"].strip() if result["msg_a"] else ""
        msg_b = result["msg_b"].strip() if result["msg_b"] else ""

        print(f"  Game {g+1:3d}: A={a_pts:+4d}  B={b_pts:+4d}  {winner_str}"
              f"  ({result['reason'].name})"
              + (f"\n           A: {msg_a}" if msg_a else "")
              + (f"\n           B: {msg_b}" if msg_b else ""))

    elapsed = time.perf_counter() - t0
    played = n_games - errors

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"  Results after {played} games ({errors} errors)  [{elapsed:.1f}s]")
    print("=" * 60)

    if played == 0:
        print("  No games completed.")
        return

    avg_a = total_a / played
    avg_b = total_b / played
    avg_total = (total_a + total_b) / played
    avg_delta = total_delta / played
    abs_delta = abs(avg_delta)

    mean = avg_total
    std_dev = (
        math.sqrt(sum((x - mean) ** 2 for x in total_pts_list) / (played - 1))
        if played > 1 else 0.0
    )

    print(f"  {'Record':15} A: {wins_a}  B: {wins_b}  Ties: {ties}")
    if not same_agent:
        win_rate = wins_a / played * 100
        print(f"  {'A win rate':15} {win_rate:.1f}%")
    print(f"  {'avg A pts':15} {avg_a:.1f}")
    print(f"  {'avg B pts':15} {avg_b:.1f}")
    print(f"  {'avg total':15} {avg_total:.1f}")
    print(f"  {'std dev total':15} {std_dev:.1f}")
    print(f"  {'avg |delta|':15} {abs_delta:.1f}")

    if same_agent:
        print()
        print("  (Same agent self-play: win rate is noise. Focus on avg_total and std_dev.)")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    args = sys.argv[1:]
    agent_a = args[0] if len(args) >= 1 else AGENT_A
    agent_b = args[1] if len(args) >= 2 else AGENT_B
    n_games = int(args[2]) if len(args) >= 3 else NUM_GAMES

    main(agent_a, agent_b, n_games)