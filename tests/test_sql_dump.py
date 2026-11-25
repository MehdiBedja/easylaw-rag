from easylaw_rag.ingest.sql_dump import parse_values
from easylaw_rag.text.arabic import normalize


def test_parse_basic_row():
    rows = list(parse_values("(1,'abc',NULL,-2.5),(2,'x',3,0);"))
    assert rows == [(1, "abc", None, -2.5), (2, "x", 3, 0)]


def test_parse_escapes_and_parentheses_inside_strings():
    rows = list(parse_values(r"(1,'it\'s (fine), ok\n2','a''b','back\\slash');"))
    assert rows == [(1, "it's (fine), ok\n2", "a'b", "back\\slash")]


def test_parse_arabic_and_zero_padded_ids():
    rows = list(parse_values("(776,'استدراك في','000265','065602');"))
    assert rows == [(776, "استدراك في", "000265", "065602")]


def test_normalize_arabic():
    assert normalize("أَمْرٌ  إِلى‏ المادّة") == "امر الي الماده"
    assert normalize("سنة ١٩٨٤") == "سنه 1984"
