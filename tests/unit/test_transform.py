import pandas as pd
import pytest

from wildfire_platform.batch.transform import transform


def row(**overrides):
    base = {
        "latitude": 33.76,
        "longitude": -7.43,
        "acq_date": "2025-07-01",
        "acq_time": "0251",
        "satellite": "N",
        "instrument": "VIIRS",
        "confidence": "n",
        "frp": 1.4,
        "daynight": "N",
        "type": 0,
    }
    base.update(overrides)
    return base


def test_datetime_is_built_with_zero_padded_time():
    out = transform(pd.DataFrame([row(acq_time="0251"), row(acq_time="1249", latitude=30.0)]))
    assert out.loc[0, "acq_datetime"] == pd.Timestamp("2025-07-01 02:51")
    assert out.loc[1, "acq_datetime"] == pd.Timestamp("2025-07-01 12:49")
    assert list(out["hour"]) == [2, 12]


def test_time_parts_are_derived():
    out = transform(pd.DataFrame([row()]))
    assert out.loc[0, "year"] == 2025
    assert out.loc[0, "month"] == 7


def test_grid_cell_negative_longitude():
    # 33.76 -> 33.50 and -7.43 -> -7.50 with 0.5 degree cells
    out = transform(pd.DataFrame([row()]), cell_size=0.5)
    assert out.loc[0, "cell_id"] == "33.50_-7.50"


def test_grid_cell_on_boundary_belongs_to_upper_cell():
    out = transform(pd.DataFrame([row(latitude=34.0, longitude=-7.0)]), cell_size=0.5)
    assert out.loc[0, "cell_id"] == "34.00_-7.00"


def test_invalid_cell_size_raises():
    with pytest.raises(ValueError):
        transform(pd.DataFrame([row()]), cell_size=0)


def test_confidence_labels():
    df = pd.DataFrame([row(confidence="l"), row(confidence="n", latitude=30.0), row(confidence="h", latitude=31.0), row(confidence="x", latitude=32.0)])
    out = transform(df)
    assert list(out["confidence_label"]) == ["low", "nominal", "high", "unknown"]


def test_vegetation_fire_flag_keeps_all_rows():
    df = pd.DataFrame([row(type=0), row(type=2, latitude=30.0)])
    out = transform(df)
    assert len(out) == 2
    assert list(out["is_presumed_vegetation_fire"]) == [True, False]


def test_missing_optional_columns_do_not_crash():
    df = pd.DataFrame([row()]).drop(columns=["type", "frp", "confidence"])
    out = transform(df)
    assert out["is_presumed_vegetation_fire"].isna().all()
    assert out.loc[0, "confidence_label"] == "unknown"


def test_detection_id_is_deterministic_and_unique():
    df = pd.DataFrame([row(), row(latitude=30.0)])
    ids_1 = transform(df)["detection_id"]
    ids_2 = transform(df)["detection_id"]
    assert list(ids_1) == list(ids_2)
    assert ids_1.nunique() == 2


def test_input_is_not_modified():
    df = pd.DataFrame([row()])
    before = df.copy()
    transform(df)
    pd.testing.assert_frame_equal(df, before)