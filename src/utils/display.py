"""Pretty CLI display helpers for debate progress."""

from __future__ import annotations

from src.schemas.models import CulturalRating, DebateRound

# ANSI colors
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"
_BLUE = "\033[94m"


def _score_color(score: int) -> str:
    if score >= 81:
        return _GREEN
    if score >= 61:
        return _CYAN
    if score >= 41:
        return _YELLOW
    return _RED


def print_header(title: str, char: str = "═", width: int = 72):
    line = char * width
    print(f"\n{_BOLD}{line}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{line}{_RESET}")


def print_story_banner(story_id: str, culture: str, config_id: str):
    print(f"\n{_BOLD}{'#' * 72}{_RESET}")
    print(f"{_BOLD}#  STORY: {story_id}  |  Culture: {culture}  |  Config: {config_id}{_RESET}")
    print(f"{_BOLD}{'#' * 72}{_RESET}")


def print_ratings_table(ratings: list[CulturalRating], phase: str, spread: int, threshold: int):
    scores = [r.score for r in ratings]
    avg = sum(scores) / len(scores) if scores else 0

    print(f"\n{_BOLD}  {phase}{_RESET}  "
          f"{_DIM}(spread={spread}, threshold={threshold}, avg={avg:.0f}){_RESET}")
    print(f"  {'─' * 66}")

    for r in ratings:
        sc = _score_color(r.score)
        bar_len = r.score // 2
        bar = "█" * bar_len + "░" * (50 - bar_len)
        markers_count = len(r.cultural_markers_found)
        print(
            f"  {r.agent_name:<20s} {sc}{r.score:>3d}{_RESET}  "
            f"{_DIM}{bar}{_RESET}  "
            f"{_DIM}markers={markers_count}{_RESET}"
        )

    print(f"  {'─' * 66}")


def print_debate_round_detail(rnd: DebateRound, show_critiques: bool = True, show_evidence: bool = True):
    """Print detailed debate round info."""
    for r in rnd.ratings:
        sc = _score_color(r.score)
        print(f"\n  {_BOLD}{r.agent_name}{_RESET} → {sc}{r.score}{_RESET}  {_DIM}{r.bracket}{_RESET}")

        # Show reasoning summary (first 200 chars)
        reasoning_preview = r.reasoning[:200].replace("\n", " ")
        if len(r.reasoning) > 200:
            reasoning_preview += "…"
        print(f"    {_DIM}Reasoning: {reasoning_preview}{_RESET}")

        # Show evidence
        if show_evidence and r.evidence:
            print(f"    {_BLUE}Evidence ({len(r.evidence)} items):{_RESET}")
            for ev in r.evidence[:3]:
                print(f"      • {_DIM}{ev[:120]}{_RESET}")

    # Show critiques
    if show_critiques and rnd.critiques:
        has_critiques = any(crits for crits in rnd.critiques.values())
        if has_critiques:
            print(f"\n  {_MAGENTA}Critiques:{_RESET}")
            for agent_name, crits in rnd.critiques.items():
                for c in crits[:2]:
                    print(f"    {_DIM}{agent_name} → {c[:140]}{_RESET}")


def print_consensus(reached: bool, final_score: float, total_rounds: int):
    if reached:
        print(f"\n  {_GREEN}{_BOLD}✅ CONSENSUS{_RESET}  "
              f"score={_GREEN}{final_score:.1f}{_RESET}  rounds={total_rounds}")
    else:
        print(f"\n  {_RED}{_BOLD}❌ NO CONSENSUS{_RESET}  "
              f"score={_YELLOW}{final_score:.1f}{_RESET}  rounds={total_rounds}")


def print_final_summary(results: list):
    """Print a summary table at the end of the pipeline."""
    print(f"\n{'=' * 72}")
    print(f"{_BOLD}  FINAL RESULTS SUMMARY{_RESET}")
    print(f"{'=' * 72}")
    print(f"  {'Story':<20s} {'Culture':<15s} {'Score':>6s} {'Consensus':>10s} {'Rounds':>7s}")
    print(f"  {'─' * 62}")
    for r in results:
        sc = _score_color(int(r.final_score))
        cons = f"{_GREEN}Yes{_RESET}" if r.consensus_reached else f"{_RED}No{_RESET}"
        print(
            f"  {r.story.id:<20s} {r.story.culture:<15s} "
            f"{sc}{r.final_score:>5.1f}{_RESET}  {cons:>18s}  {r.total_rounds:>5d}"
        )
    print(f"  {'─' * 62}\n")
