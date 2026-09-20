"""Deterministic mini-leagues and playoffs distinguish the revised rules."""
import numpy as np
import pytest
from seriea.simulation.rules import rank_head_to_head, resolve_playoffs

def test_head_to_head_overrides_overall_goal_difference():
    points=np.array([[10,10,4]])
    hp=np.zeros((1,3,3),dtype=int); hp[0,1,0]=6
    order=rank_head_to_head(points,np.array([[30,0,1]]),np.array([[40,10,1]]),
        hp,np.zeros_like(hp),np.random.default_rng(1))
    assert order.tolist()==[[1,0,2]]

def test_multi_team_minileague_uses_all_tied_clubs():
    points=np.array([[10,10,10,0]])
    hp=np.zeros((1,4,4),dtype=int)
    hp[0,0,1]=6; hp[0,2,0]=6;hp[0,2,1]=3
    order=rank_head_to_head(points,np.zeros((1,4)),np.zeros((1,4)),
        hp,np.zeros_like(hp),np.random.default_rng(1))
    assert order.tolist()==[[2,0,1,3]]

def test_two_leg_survival_uses_aggregate_goals_and_no_away_goals_rule():
    order=np.arange(20)[None,:]
    points=np.arange(60,0,-3)[None,:];points[0,17]=points[0,16]
    grids=np.zeros((20,20,4,4));grids[:,:,0,0]=1
    grids[16,17]=0;grids[16,17,1,0]=1
    grids[17,16]=0;grids[17,16,3,0]=1
    final,counts=resolve_playoffs(order,points,2025,grids,None,np.random.default_rng(2))
    assert final[0,16]==17
    assert counts["survival_playoffs"]==1
    assert counts["title_playoffs"]==0

def test_no_playoff_before_2022():
    order=np.arange(20)[None,:]
    points=np.zeros((1,20))
    final,counts=resolve_playoffs(order.copy(),points,2021,None,None,np.random.default_rng(2))
    assert np.array_equal(final,order)
    assert counts["survival_playoffs"]==0

def test_2022_uses_neutral_single_leg():
    order=np.arange(20)[None,:]
    points=np.arange(60,0,-3)[None,:];points[0,17]=points[0,16]
    home=np.zeros((20,20,3,3));home[:,:,2,0]=1
    neutral=np.zeros_like(home);neutral[:,:,0,2]=1
    final,_=resolve_playoffs(order,points,2022,home,neutral,np.random.default_rng(2))
    assert final[0,16]==17

def test_simulator_rejects_negative_probability_mass():
    from seriea.simulation.season import sample_scorelines
    bad=np.array([[[1.1,-.1],[0.,0.]]])
    with pytest.raises(ValueError,match="nonnegative"):
        sample_scorelines(bad,10,np.random.default_rng(1))

def test_tied_aggregate_is_resolved_by_penalties_not_away_goals():
    n=2000
    order=np.tile(np.arange(20),(n,1))
    points=np.tile(np.arange(60,0,-3),(n,1));points[:,17]=points[:,16]
    grids=np.zeros((20,20,4,4));grids[:,:,0,0]=1
    grids[16,17]=0;grids[16,17,1,0]=1
    grids[17,16]=0;grids[17,16,2,1]=1
    final,_=resolve_playoffs(order,points,2025,grids,None,np.random.default_rng(2))
    # Aggregate 2-2; first club scored away, but receives no away-goals advantage.
    assert .45<(final[:,16]==16).mean()<.55
