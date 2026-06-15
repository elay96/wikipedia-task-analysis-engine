import numpy as np
import networkx as nx
import features


def test_step_distances_of_orthogonal_then_same():
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    seq = [e1, e2, e2]  # dist(e1,e2)=1, dist(e2,e2)=0
    mean_d, var_d = features.step_distances(seq)
    assert abs(mean_d - 0.5) < 1e-9
    assert var_d > 0


def test_step_distances_too_short_is_nan():
    mean_d, var_d = features.step_distances([np.array([1.0, 0.0])])
    assert np.isnan(mean_d) and np.isnan(var_d)


def test_structural_features_counts_and_revisit():
    visits = [
        {"article": "A", "dwell_ms": 1000},
        {"article": "B", "dwell_ms": 2000},
        {"article": "A", "dwell_ms": 500},  # revisit
    ]
    f = features.structural_features(visits, n_searches=2)
    assert f["n_pages"] == 3
    assert f["n_unique_pages"] == 2
    assert abs(f["revisit_rate"] - (1 / 3)) < 1e-9
    assert f["n_searches"] == 2
    assert f["mean_dwell"] > 0


def test_step_distance_series_length_and_values():
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    series = features.step_distance_series([e1, e2, e2])
    assert len(series) == 2
    assert abs(series[0] - 1.0) < 1e-9  # orthogonal
    assert abs(series[1] - 0.0) < 1e-9  # identical


def test_dynamics_too_few_steps_is_nan():
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    f = features.dynamics_features([e1, e2])  # only 1 step
    assert np.isnan(f["dyn_slope"]) and np.isnan(f["dyn_early_late_delta"])
    assert f["dyn_n_steps"] == 1


def test_dynamics_converging_session_has_negative_slope():
    # Build vectors whose consecutive cosine distance shrinks over time.
    import math
    angles = [0.0, 1.4, 2.4, 3.0, 3.3]  # gaps: 1.4, 1.0, 0.6, 0.3 -> shrinking
    vecs = [np.array([math.cos(a), math.sin(a)]) for a in angles]
    f = features.dynamics_features(vecs)
    assert f["dyn_n_steps"] == 4
    assert f["dyn_slope"] < 0  # exploit / converging
    assert f["dyn_early_late_delta"] < 0  # later steps shorter


def test_visited_subgraph_edges_only_between_visited():
    outlinks = {"A": ["B", "Z"], "B": ["A"], "C": ["A"]}
    G = features.visited_subgraph(["A", "B", "C"], outlinks)
    assert set(G.nodes()) == {"A", "B", "C"}
    assert G.has_edge("A", "B")
    assert G.has_edge("A", "C")  # C->A counts as undirected edge
    assert not G.has_edge("A", "Z")  # Z not visited
