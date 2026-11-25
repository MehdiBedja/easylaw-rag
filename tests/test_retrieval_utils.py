from easylaw_rag.retrieval.retriever import expand_query


def test_expand_query_adds_reversed_law_number():
    assert expand_query("ما هو القانون 84-17") == "ما هو القانون 84-17 17-84"


def test_expand_query_leaves_plain_text():
    assert expand_query("شروط الطلاق") == "شروط الطلاق"
