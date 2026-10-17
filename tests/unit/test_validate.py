import pandas as pd
import pytest

from wildfire_platform.batch.validate import ValidationError, validate


def row(**overrides):
    """One valid fire detection; tests change only the field they care about."""
    base = {
        "latitude": 33.5,
        "longitude": -7.5,
        "acq_date": "2025-07-01",
        "acq_time": "1249",
        "confidence": "n",
        "frp": 1.4,
    }
    base.update(overrides)
    return base


def test_valid_input_passes():
    df = pd.DataFrame([row(), row(latitude=34.0)])
    result = validate(df)
    assert result.report["rows_in"] == 2
    assert result.report["rows_clean"] == 2
    assert result.report["rows_rejected"] == 0


def test_empty_input_raises():
    with pytest.raises(ValidationError, match="empty"):
        validate(pd.DataFrame())


def test_missing_required_column_raises():
    df = pd.DataFrame([row()]).drop(columns=["latitude"])
    with pytest.raises(ValidationError, match="latitude"):
        validate(df)


def test_invalid_coordinates_are_rejected():
    df = pd.DataFrame([row(), row(latitude=95.0, acq_time="1300"), row(longitude=-200.0, acq_time="1400")])
    result = validate(df)
    assert result.report["rows_clean"] == 1
    assert result.report["rejected_by_reason"] == {"invalid_coordinates": 2}


def test_invalid_date_is_rejected():
    df = pd.DataFrame([row(), row(acq_date="not-a-date", acq_time="1300")])
    result = validate(df)
    assert result.report["rejected_by_reason"] == {"invalid_date": 1}


def test_invalid_time_is_rejected():
    df = pd.DataFrame([row(), row(acq_time="2575")])  # 25h 75min does not exist
    result = validate(df)
    assert result.report["rejected_by_reason"] == {"invalid_time": 1}


def test_duplicates_are_rejected_and_first_is_kept():
    df = pd.DataFrame([row(), row()])
    result = validate(df)
    assert result.report["rows_clean"] == 1
    assert result.report["rejected_by_reason"] == {"duplicate": 1}


def test_all_rows_rejected_raises():
    df = pd.DataFrame([row(latitude=999.0)])
    with pytest.raises(ValidationError, match="All rows"):
        validate(df)