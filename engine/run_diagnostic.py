"""
run_diagnostic.py - Match diagnostics
Run from distb4repo/ the same way run_local_agents.py works:

    python engine/run_diagnostic.py AgentA AgentB 5
"""

import math
import os
import pathlib
import sys
import time
import multiprocessing
from typing import Optional

DEFAULT_AGENT_A = "Design_2_Kai"
DEFAULT_AGENT_B = "Design_2_Kai"
DEFAULT_N_GAMES = 5
DETAIL_GAMES    = 2


def _find_engine_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).parent.resolve()
    if (here / "run_local_agents.py").exists():
        return here
    raise FileNotFoundError("Run from distb4repo/: python engine/run_diagnostic.py ...")


def _run_one_game(engine_dir: pathlib.Path, agent_a: str, agent_b: str) -> Optional[dict]:
    # Exact same path logic as run_local_agents.py
    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(str(top_level), "3600-agents")

    engine_str = str(engine_dir)
    for p in [engine_str, play_directory]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from gameplay import play_game

    try:
        board, rat_hist, spawn_a, spawn_b, msg_a, msg_b = play_game(
            play_directory, play_directory,
            agent_a, agent_b,
            display_game=False,
            delay=0.0,
            clear_screen=False,
            record=True,
            limit_resources=False,
        )
    except Exception as e:
        print(f"  [ERROR] play_game: {e}")
        return None

    # Pull history off the board directly - no get_history_dict
    from game.enums import ResultArbiter, MoveType
    w = board.get_winner()
    winner = (
        "A" if w == ResultArbiter.PLAYER_A else
        "B" if w == ResultArbiter.PLAYER_B else
        "TIE"
    )

    if not board.build_history or board.history is None:
        print("  [ERROR] No history recorded.")
        return None

    hist = board.history

    # left_behind_enums is a list of MoveType per turn
    left = hist.left_behind_enums
    rat  = hist.rat_caught
    a_points_hist = [0] + hist.a_points
    b_points_hist = [0] + hist.b_points

    # A plays turns 0,2,4... B plays turns 1,3,5...
    a_moves = [left[i] for i in range(0, len(left), 2)]
    b_moves = [left[i] for i in range(1, len(left), 2)]

    a_catches = sum(1 for i in range(0, len(rat), 2) if rat[i])
    b_catches = sum(1 for i in range(1, len(rat), 2) if rat[i])

    # Carpet roll lengths from new_carpets via pos history
    # We can infer roll length from point jumps on carpet turns
    CARPET_PTS = {-1:1, 2:2, 4:3, 6:4, 10:5, 15:6, 21:7}
    a_rolls, b_rolls = [], []
    a_carpet_pts = 0
    b_carpet_pts = 0
    for i, move in enumerate(left):
        if move == MoveType.CARPET:
            if i % 2 == 0:
                # A's turn - index into a_points_hist
                a_idx = i // 2 + 1
                if a_idx < len(a_points_hist):
                    delta = a_points_hist[a_idx] - a_points_hist[a_idx - 1]
                    a_carpet_pts += delta
                    roll = CARPET_PTS.get(delta, 0)
                    if roll: a_rolls.append(roll)
            else:
                b_idx = i // 2 + 1
                if b_idx < len(b_points_hist):
                    delta = b_points_hist[b_idx] - b_points_hist[b_idx - 1]
                    b_carpet_pts += delta
                    roll = CARPET_PTS.get(delta, 0)
                    if roll: b_rolls.append(roll)

    def max_dry(moves):
        best = cur = 0
        for m in moves:
            if m != MoveType.CARPET:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        return best

    # Turn-by-turn trajectory
    trajectory = []
    n_rounds = min(len(a_moves), len(b_moves))
    for r in range(n_rounds):
        a_rat = rat[r*2] if r*2 < len(rat) else False
        b_rat = rat[r*2+1] if r*2+1 < len(rat) else False
        trajectory.append({
            "round": r+1,
            "a_pts": a_points_hist[r+1] if r+1 < len(a_points_hist) else 0,
            "b_pts": b_points_hist[r+1] if r+1 < len(b_points_hist) else 0,
            "a_move": a_moves[r].name,
            "b_move": b_moves[r].name,
            "a_rat": bool(a_rat),
            "b_rat": bool(b_rat),
        })

    return {
        "winner":       winner,
        "reason":       board.win_reason.name,
        "a_final":      a_points_hist[-1],
        "b_final":      b_points_hist[-1],
        "a_searches":   sum(1 for m in a_moves if m == MoveType.SEARCH),
        "b_searches":   sum(1 for m in b_moves if m == MoveType.SEARCH),
        "a_carpets":    sum(1 for m in a_moves if m == MoveType.CARPET),
        "b_carpets":    sum(1 for m in b_moves if m == MoveType.CARPET),
        "a_primes":     sum(1 for m in a_moves if m == MoveType.PRIME),
        "b_primes":     sum(1 for m in b_moves if m == MoveType.PRIME),
        "a_plains":     sum(1 for m in a_moves if m == MoveType.PLAIN),
        "b_plains":     sum(1 for m in b_moves if m == MoveType.PLAIN),
        "a_catches":    a_catches,
        "b_catches":    b_catches,
        "a_rolls":      a_rolls,
        "b_rolls":      b_rolls,
        "a_carpet_pts": a_carpet_pts,
        "b_carpet_pts": b_carpet_pts,
        "a_max_dry":    max_dry(a_moves),
        "b_max_dry":    max_dry(b_moves),
        "trajectory":   trajectory,
    }


def _print_game(g: dict, game_num: int, show_trajectory: bool):
    print(f"\n  Game {game_num}: A={g['a_final']:+d}  B={g['b_final']:+d}  -> {g['winner']} wins ({g['reason']})")
    print(f"  {'':14} {'A':>5} {'B':>5}")
    for label, ak, bk in [
        ("Searches",   "a_searches",   "b_searches"),
        ("Catches",    "a_catches",    "b_catches"),
        ("Carpets",    "a_carpets",    "b_carpets"),
        ("Primes",     "a_primes",     "b_primes"),
        ("Plains",     "a_plains",     "b_plains"),
        ("CarpetPts",  "a_carpet_pts", "b_carpet_pts"),
        ("MaxDryStrk", "a_max_dry",    "b_max_dry"),
    ]:
        print(f"  {label:14} {g[ak]:>5} {g[bk]:>5}")
    print(f"  Rolls A: {sorted(g['a_rolls'])}")
    print(f"  Rolls B: {sorted(g['b_rolls'])}")

    if show_trajectory:
        print(f"\n  Turn-by-turn:")
        print(f"  {'Rnd':>3}  {'A pts':>6}  {'B pts':>6}  {'diff':>5}  {'A move':<12}  {'B move':<12}")
        print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*12}  {'-'*12}")
        for t in g["trajectory"]:
            a_tag = " R" if t["a_rat"] else ""
            b_tag = " R" if t["b_rat"] else ""
            print(f"  {t['round']:>3}  {t['a_pts']:>6}  {t['b_pts']:>6}  "
                  f"{t['a_pts']-t['b_pts']:>+5}  "
                  f"{t['a_move']+a_tag:<14}  {t['b_move']+b_tag:<14}")


def _print_aggregate(games: list, agent_a: str, agent_b: str):
    n = len(games)
    if n == 0:
        print("No completed games.")
        return

    def avg(vals): return sum(vals)/len(vals) if vals else 0.0
    def std(vals):
        if len(vals) < 2: return 0.0
        m = avg(vals)
        return math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1))

    totals = [g["a_final"]+g["b_final"] for g in games]
    deltas = [abs(g["a_final"]-g["b_final"]) for g in games]
    a_wins = sum(1 for g in games if g["winner"]=="A")
    b_wins = sum(1 for g in games if g["winner"]=="B")
    ties   = sum(1 for g in games if g["winner"]=="TIE")

    print("\n" + "="*55)
    print(f"AGGREGATE  {n} games   {agent_a} (A) vs {agent_b} (B)")
    print("="*55)
    print(f"  Record:      A={a_wins}  B={b_wins}  Ties={ties}")
    print(f"  Avg pts:     A={avg([g['a_final'] for g in games]):.1f}  B={avg([g['b_final'] for g in games]):.1f}")
    print(f"  Avg total:   {avg(totals):.1f}  (std={std(totals):.1f})")
    print(f"  Avg |delta|: {avg(deltas):.1f}")
    print()
    print(f"  {'Metric':<14} {'avg A':>6} {'avg B':>6}")
    print(f"  {'-'*14} {'-'*6} {'-'*6}")
    for label, ak, bk in [
        ("Searches",   "a_searches",   "b_searches"),
        ("Catches",    "a_catches",    "b_catches"),
        ("Carpets",    "a_carpets",    "b_carpets"),
        ("Primes",     "a_primes",     "b_primes"),
        ("Plains",     "a_plains",     "b_plains"),
        ("CarpetPts",  "a_carpet_pts", "b_carpet_pts"),
        ("MaxDryStrk", "a_max_dry",    "b_max_dry"),
    ]:
        print(f"  {label:<14} {avg([g[ak] for g in games]):>6.1f} {avg([g[bk] for g in games]):>6.1f}")

    all_a = [r for g in games for r in g["a_rolls"]]
    all_b = [r for g in games for r in g["b_rolls"]]
    print()
    print(f"  All A rolls: {sorted(all_a)}")
    print(f"  All B rolls: {sorted(all_b)}")
    if all_a:
        print(f"  Avg roll:    A={avg(all_a):.2f}" + (f"  B={avg(all_b):.2f}" if all_b else ""))
    print("="*55)


def main():
    agent_a = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AGENT_A
    agent_b = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_AGENT_B
    n_games = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_GAMES

    engine_dir = _find_engine_dir()
    print(f"Agent A : {agent_a}")
    print(f"Agent B : {agent_b}")
    print(f"Games   : {n_games}")

    t0 = time.perf_counter()
    completed = []

    for g in range(n_games):
        print(f"\n--- Game {g+1}/{n_games} ---", flush=True)
        result = _run_one_game(engine_dir, agent_a, agent_b)
        if result is None:
            print("  [SKIPPED]")
            continue
        completed.append(result)
        _print_game(result, g+1, show_trajectory=(g < DETAIL_GAMES))

    print(f"\n[{time.perf_counter()-t0:.1f}s elapsed]")
    _print_aggregate(completed, agent_a, agent_b)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()