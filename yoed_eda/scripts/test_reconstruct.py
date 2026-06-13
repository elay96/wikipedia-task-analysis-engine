import reconstruct


def test_epoch_ms_to_iso_utc():
    # 1764411186897 ms -> 2025-11-29T...Z (UTC, second precision is fine for rvstart)
    iso = reconstruct.epoch_ms_to_iso(1764411186897)
    assert iso.startswith("2025-11-29T")
    assert iso.endswith("Z")


def test_epoch_ms_to_iso_handles_nan():
    assert reconstruct.epoch_ms_to_iso(float("nan")) is None


def test_cache_path_uses_revid():
    p = reconstruct.cache_path("/tmp/cache", 123456)
    assert str(p).endswith("123456.json")
