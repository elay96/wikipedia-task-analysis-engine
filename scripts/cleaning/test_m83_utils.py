from m83_utils import title_to_slug, slug_to_title


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
