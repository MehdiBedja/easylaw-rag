from easylaw_rag.ingest.joradp import journal_url, parse_years


def test_journal_url():
    assert journal_url("ar", 1970, 5) == "http://www.joradp.dz/FTP/JO-ARABE/1970/A1970005.pdf"
    assert journal_url("fr", 2005, 15) == "http://www.joradp.dz/FTP/JO-FRANCAIS/2005/F2005015.pdf"


def test_parse_years():
    assert parse_years("1970-1973,1991") == {1970, 1971, 1972, 1973, 1991}
    assert parse_years(None) is None
