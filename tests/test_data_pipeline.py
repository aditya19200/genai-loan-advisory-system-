from data.processing import DataProcessor


def test_prepare_data_has_target_and_sensitive_attrs():
    processor = DataProcessor("data/german_credit.csv")
    prepared = processor.prepare_data()
    assert len(prepared.X_train) > 0
    assert "Sex" in prepared.sensitive_train.columns
    assert "age_group" in prepared.sensitive_train.columns
