"""Serie A ranking and playoffs, with explicit season and probability inputs.

Sources: FIGC 304/A (2022/23), FIGC decision 7 July 2023, and FIGC 331/A
(2025/26), reproduced on p.44 of the Lega 2025/26 regulations.
Ranking: points, head-to-head points, head-to-head goal difference, overall
goal difference, goals scored, drawing lots. No recursive mini-table restart.
Playoffs: from 2022/23 at first/second and seventeenth/eighteenth when tied;
2022/23 neutral single match; from 2023/24 title at better-ranked club,
survival over two legs. No extra time or away-goals rule; penalties modeled 50/50.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def rank_head_to_head(points, goal_difference, goals_for, h2h_points, h2h_gd, rng):
    """Rank each simulation, recomputing the mini-table for each points-tied group."""
    n, teams = points.shape
    order = np.argsort(-points, axis=1, kind="stable")
    for simulation in range(n):
        ordered = order[simulation]
        sorted_points = points[simulation, ordered]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_points)) + 1, teams]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end-start < 2:
                continue
            group = ordered[start:end].copy()
            hp = h2h_points[simulation][np.ix_(group, group)].sum(axis=1)
            gd = h2h_gd[simulation][np.ix_(group, group)].sum(axis=1)
            key_order = np.lexsort((rng.random(len(group)), -goals_for[simulation,group],
                -goal_difference[simulation,group], -gd, -hp))
            ordered[start:end] = group[key_order]
    return order

def resolve_playoffs(order, points, season_start, grids, neutral_grids, rng):
    """Swap only playoff participants; return trigger counts and resolved ordering."""
    from seriea.simulation.season import sample_scorelines
    n, teams = points.shape
    diagnostics = {"title_playoffs":0, "survival_playoffs":0}
    if season_start < 2022:
        return order, diagnostics
    if teams != 20:
        raise ValueError("Serie A playoffs require a 20-team table.")
    if grids is None:
        raise ValueError("Playoff score grids are required for seasons from 2022/23.")
    for boundary, label in ((0,"title"),(16,"survival")):
        better = order[:,boundary].copy()
        worse = order[:,boundary+1].copy()
        tied = points[np.arange(n),better] == points[np.arange(n),worse]
        rows = np.flatnonzero(tied)
        diagnostics[label+"_playoffs"] = len(rows)
        # Batch simulations with the same ordered participants.
        for first, second in np.unique(np.column_stack([better[rows],worse[rows]]),axis=0):
            selected = rows[(better[rows]==first)&(worse[rows]==second)]
            if season_start == 2022:
                if neutral_grids is None:
                    raise ValueError("2022/23 needs neutral-venue playoff distributions.")
                hg,ag=sample_scorelines(neutral_grids[first,second][None],len(selected),rng)
                margin=(hg-ag)[:,0]
            else:
                # The better club's home leg is the return leg for survival.
                # Independent fixed-rate draws can be generated in either order.
                hg,ag=sample_scorelines(grids[first,second][None],len(selected),rng)
                margin=(hg-ag)[:,0]
                if label=="survival":
                    # Other leg at the lower-ranked club; aggregate goals only.
                    hg,ag=sample_scorelines(grids[second,first][None],len(selected),rng)
                    margin+=(ag-hg)[:,0]
            first_wins=(margin>0)|((margin==0)&(rng.random(len(selected))<.5))
            swap=selected[~first_wins]
            order[swap,boundary]=second
            order[swap,boundary+1]=first
    return order, diagnostics

def playoff_score_grids(model, teams, *, neutral=False):
    """Build all possible playoff pairings with fixed pre-origin model ratings."""
    import copy
    fitted=copy.copy(model)
    if neutral:
        fitted._home_advantage=0.
    fixtures=pd.DataFrame([(h,a) for h in teams for a in teams],columns=["home","away"])
    g=fitted.predict_score_grid(fixtures)
    return g.reshape(len(teams),len(teams),g.shape[1],g.shape[2])
