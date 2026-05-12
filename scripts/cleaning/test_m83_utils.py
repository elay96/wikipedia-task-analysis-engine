import networkx as nx
from m83_utils import title_to_slug, slug_to_title, network_metrics


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
