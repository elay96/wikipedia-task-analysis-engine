import networkx as nx
import numpy as np
import pandas as pd
from m83_utils import (
    title_to_slug,
    slug_to_title,
    network_metrics,
    forward_flow,
    bh_score,
    fdr_bh,
    collect_diffuse_slugs,
)


class TestTitleSlugRoundtrip:
    def test_title_with_spaces_to_slug(self):
        assert title_to_slug("Aquatic plant") == "Aquatic_plant"

    def test_slug_to_title(self):
        assert slug_to_title("Modern_art") == "Modern art"

    def test_roundtrip_preserves_value(self):
        assert title_to_slug(slug_to_title("Brain_Structure_and_Function")) == \
               "Brain_Structure_and_Function"

    def test_anchor_is_stripped(self):
        assert title_to_slug("Standard deviation#Table") == "Standard_deviation"


class TestNetworkMetrics:
    def test_triangle_is_dense_and_clustered(self):
        G = nx.complete_graph(3)
        m = network_metrics(G)
        assert m["n_unique"] == 3
        assert m["n_edges"] == 3
        assert m["clustering"] == 1.0
        assert m["char_path_length"] == 1.0
        assert m["global_efficiency"] == 1.0
        assert m["lcc_fraction"] == 1.0

    def test_path_graph_has_no_clustering(self):
        G = nx.path_graph(4)
        m = network_metrics(G)
        assert m["n_unique"] == 4
        assert m["n_edges"] == 3
        assert m["clustering"] == 0.0
        assert abs(m["char_path_length"] - 10 / 6) < 1e-9
        assert m["lcc_fraction"] == 1.0

    def test_disconnected_uses_largest_component(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_node(99)
        m = network_metrics(G)
        assert m["n_unique"] == 4
        assert m["n_edges"] == 3
        assert m["char_path_length"] == 1.0
        assert abs(m["lcc_fraction"] - 0.75) < 1e-9

    def test_single_node_returns_zeroed_metrics(self):
        G = nx.Graph()
        G.add_node(0)
        m = network_metrics(G)
        assert m["n_unique"] == 1
        assert m["n_edges"] == 0
        assert m["clustering"] == 0.0
        assert m["char_path_length"] == 0.0
        assert m["global_efficiency"] == 0.0
        assert m["lcc_fraction"] == 1.0


class TestForwardFlow:
    def test_two_orthogonal_vectors_yield_distance_one(self):
        vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        assert abs(forward_flow(vecs) - 1.0) < 1e-9

    def test_two_identical_vectors_yield_zero(self):
        v = np.array([1.0, 0.0])
        assert forward_flow([v, v]) == 0.0

    def test_three_step_sequence_averaging(self):
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        e3 = np.array([1.0, 0.0])
        assert abs(forward_flow([e1, e2, e3]) - 0.75) < 1e-9

    def test_single_vector_returns_nan(self):
        result = forward_flow([np.array([1.0, 0.0])])
        assert np.isnan(result)

    def test_empty_sequence_returns_nan(self):
        assert np.isnan(forward_flow([]))


class TestBHScore:
    def test_constant_input_yields_zero(self):
        df = pd.DataFrame({
            "n_edges": [10, 10, 10],
            "clustering": [0.3, 0.3, 0.3],
            "global_efficiency": [0.5, 0.5, 0.5],
            "char_path_length": [2.0, 2.0, 2.0],
        })
        scores = bh_score(df)
        assert all(abs(s) < 1e-9 for s in scores)

    def test_higher_edges_clustering_eff_pushes_hunter_positive(self):
        df = pd.DataFrame({
            "n_edges":             [1,   5,   10],
            "clustering":          [0.1, 0.3, 0.5],
            "global_efficiency":   [0.1, 0.3, 0.5],
            "char_path_length":    [3.0, 2.0, 1.0],
        })
        scores = bh_score(df)
        assert scores[2] > scores[1] > scores[0]
        assert abs(sum(scores)) < 1e-9

    def test_higher_path_alone_pushes_busybody_negative(self):
        df = pd.DataFrame({
            "n_edges":           [5, 5, 5],
            "clustering":        [0.3, 0.3, 0.3],
            "global_efficiency": [0.4, 0.4, 0.4],
            "char_path_length":  [1.0, 2.0, 3.0],
        })
        scores = bh_score(df)
        assert scores[2] < scores[1] < scores[0]


class TestFDRBH:
    def test_all_ones_become_ones(self):
        adj = fdr_bh([1.0, 1.0, 1.0])
        assert all(abs(a - 1.0) < 1e-9 for a in adj)

    def test_classical_example_4_pvalues(self):
        adj = fdr_bh([0.001, 0.01, 0.04, 0.5])
        assert abs(adj[0] - 0.004) < 1e-9
        assert abs(adj[1] - 0.02) < 1e-9
        assert abs(adj[2] - 0.0533333333) < 1e-7
        assert abs(adj[3] - 0.5) < 1e-9

    def test_nan_inputs_remain_nan(self):
        adj = fdr_bh([0.01, float("nan"), 0.5])
        assert abs(adj[0] - 0.02) < 1e-9
        assert np.isnan(adj[1])
        assert abs(adj[2] - 0.5) < 1e-9


class TestCollectDiffuseSlugs:
    def test_returns_unique_slugs_only_for_diffuse_article_opens(self):
        df = pd.DataFrame({
            "ID":         [1, 1, 2, 3, 3],
            "Condition":  ["diffuse", "diffuse", "clumpy", "diffuse", "diffuse"],
            "Action":     ["article_open", "article_open", "article_open",
                           "article_open", "search"],
            "ArticleSlug": ["A", "B", "C", "A", "ignored"],
        })
        slugs = collect_diffuse_slugs(df)
        assert slugs == {"A", "B"}
