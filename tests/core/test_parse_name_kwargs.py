import pytest

from nnsmith.backends.factory import parse_name_kwargs


def test_single_invalid():
    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("+")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("@something")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("something@")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("pt2@something")


def test_single_valid():
    assert parse_name_kwargs("pt2") == ("pt2", {})
    assert parse_name_kwargs("pt2 ") == ("pt2", {})
    assert parse_name_kwargs("pt2   ") == ("pt2", {})
    assert parse_name_kwargs(" pt2") == ("pt2", {})
    assert parse_name_kwargs("  pt2") == ("pt2", {})
    assert parse_name_kwargs("  pt2  ") == ("pt2", {})


def test_kwargs_invalid():
    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("pt2 foo")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("pt2 foo@")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("pt2 @bar")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("pt2 foo@bar baz")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("pt2 foo@ qux")

    with pytest.raises(ValueError, match="Invalid backend"):
        parse_name_kwargs("pt2 foo@@qux")


def test_kwargs_valid():
    assert parse_name_kwargs("pt2 foo@bar") == ("pt2", {"foo": "bar"})

    assert parse_name_kwargs("pt2 foo@bar baz@qux") == (
        "pt2",
        {"foo": "bar", "baz": "qux"},
    )

    assert parse_name_kwargs("pt2 foo@bar baz@qux quux@quuz") == (
        "pt2",
        {"foo": "bar", "baz": "qux", "quux": "quuz"},
    )

    # add a few random space
    assert parse_name_kwargs("pt2 foo@bar baz@qux  quux@quuz") == (
        "pt2",
        {"foo": "bar", "baz": "qux", "quux": "quuz"},
    )
