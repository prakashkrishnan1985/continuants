"""Tests for the customer_db mock API."""

import pytest

from src.apis.customer_db import CustomerDB, CustomerDBError


def test_lookup_returns_known_customer():
    db = CustomerDB(seed=1)
    record = db.get_customer("cust_001")
    assert record["customer_id"] == "cust_001"
    assert record["name"] == "Alex Rivera"
    assert record["tier"] == "pro"


def test_lookup_unknown_customer_raises():
    db = CustomerDB(seed=1)
    with pytest.raises(CustomerDBError, match="customer_not_found"):
        db.get_customer("cust_does_not_exist")


def test_search_by_email_finds_partial_match():
    db = CustomerDB(seed=1)
    matches = db.search_customers_by_email("priya")
    assert len(matches) == 1
    assert matches[0]["customer_id"] == "cust_002"


def test_search_by_email_no_match_returns_empty():
    db = CustomerDB(seed=1)
    assert db.search_customers_by_email("nobody-by-this-handle") == []


def test_failure_injection_eventually_fails():
    # With 100% failure rate, every call should fail.
    db = CustomerDB(failure_rate=1.0, seed=1)
    with pytest.raises(CustomerDBError):
        db.get_customer("cust_001")


def test_zero_failure_rate_never_fails():
    db = CustomerDB(failure_rate=0.0, seed=1)
    # Many calls in a row should succeed.
    for _ in range(50):
        db.get_customer("cust_001")


def test_rotation_does_not_change_identity_fields():
    # With max rotation, identifying fields must still be preserved.
    db = CustomerDB(rotation_strength=1.0, seed=1)
    for _ in range(50):
        record = db.get_customer("cust_001")
        assert record["customer_id"] == "cust_001"
        assert record["email"] == "alex.rivera@example.com"
        assert record["name"] == "Alex Rivera"
        assert record["tier"] == "pro"


def test_seed_reproducibility():
    db_a = CustomerDB(rotation_strength=0.5, failure_rate=0.1, seed=42)
    db_b = CustomerDB(rotation_strength=0.5, failure_rate=0.1, seed=42)

    # Two DBs with the same seed should produce the same sequence of
    # outcomes given the same call sequence.
    results_a = []
    results_b = []
    for _ in range(20):
        try:
            results_a.append(db_a.get_customer("cust_001"))
        except CustomerDBError as e:
            results_a.append(str(e))
        try:
            results_b.append(db_b.get_customer("cust_001"))
        except CustomerDBError as e:
            results_b.append(str(e))

    assert results_a == results_b


def test_list_customer_ids_returns_all_seed_customers():
    db = CustomerDB(seed=1)
    ids = db.list_customer_ids()
    expected = {"cust_001", "cust_002", "cust_003", "cust_004", "cust_005"}
    assert set(ids) == expected
