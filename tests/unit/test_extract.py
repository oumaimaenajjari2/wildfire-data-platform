import pytest

from wildfire_platform.batch.extract import ExtractError, list_raw_files, read_raw, sha256_of


def test_sha256_is_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x")
    b.write_text("y")
    assert sha256_of(a) == sha256_of(a)
    assert sha256_of(a) != sha256_of(b)


def test_list_raw_files_missing_dir_raises(tmp_path):
    with pytest.raises(ExtractError):
        list_raw_files(tmp_path / "nope")


def test_read_raw_keeps_leading_zero_in_time(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("latitude,acq_time\n33.5,0251\n")
    assert read_raw(p).loc[0, "acq_time"] == "0251"