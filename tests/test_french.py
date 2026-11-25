from easylaw_rag.ingest.joradp_search import parse_page
from easylaw_rag.ingest.pdf_pages import layout_text
from easylaw_rag.process.chunk import article_label, chunk_paragraphs
from easylaw_rag.process.metadata import citation_fr, is_administrative, is_administrative_fr
from easylaw_rag.process.segment import segment
from easylaw_rag.text.clean import to_lines, to_paragraphs
from easylaw_rag.text.lang import FR

PAGE = """4 JOURNAL OFFICIEL DE LA REPUBLIQUE ALGERIENNE N° 15 27 février 2005
Art. 9. — Le présent arrêté sera publié au Journal
officiel de la République algérienne démocratique et populaire.
Fait à Alger, le 3 mars 2005.
MINISTERE DES FINANCES
Décret exécutif n° 05-67 du 19 Dhou El Hidja 1425
correspondant au 30 janvier 2005 portant création du
comité national du Codex Alimentarius et fixant ses missions.
————
Le Chef du Gouvernement,
Vu la Constitution, notamment ses articles 85-4° et
125 ;
Vu le
décret exécutif n° 90-39 du 30 janvier 1990 relatif au contrôle de la qualité ;
Décrète :
Article 1er. — Il est créé un comité national du Codex
Alimentarius, ci-après dénommé « le comité ».
Art. 2. — Le comité est chargé de coordonner les tra-
vaux relatifs aux normes alimentaires.
Fait à Alger, le 19 Dhou El Hidja 1425 correspondant au
30 janvier 2005.
Ahmed OUYAHIA.
Arrêté du 15 février 2005 portant délégation de signature au directeur du budget.
"""
DESCRIPTION = "Portant création du comité national du Codex Alimentarius et fixant ses missions et son organisation."


def body_of(page: str = PAGE) -> str:
    lines = to_lines(page, FR)
    seg = segment(lines, DESCRIPTION, FR)
    assert seg.title_found
    return "\n".join(p.text for p in to_paragraphs(lines[seg.start : seg.end], FR))


def test_banner_is_noise_but_sentence_is_kept():
    lines = to_lines(PAGE, FR)
    assert lines[0] is None  # page banner with the issue number
    assert lines[2] is not None  # "officiel de la République algérienne démocratique ..." inside a sentence
    assert to_lines("Journal officiel de la République algérienne démocratique et populaire.", FR)[0] is not None
    assert to_lines("————", FR)[0] is None


def test_segment_cuts_previous_and_next_text():
    body = body_of()
    assert body.startswith("Décret exécutif n° 05-67")
    assert "Ahmed OUYAHIA" in body
    assert "30 janvier 2005." in body  # a date alone on a line is the end of "Fait à Alger, le ..."
    assert "125 ;" in body  # short continuation lines survive
    assert "MINISTERE DES FINANCES" not in body
    assert "délégation de signature" not in body


def test_lowercase_reference_in_a_visa_is_not_a_header():
    # "décret exécutif n° 90-39 ... relatif ..." starts a line but is a quotation inside a visa
    assert "Article 1er" in body_of()


def test_articles_and_hyphenation():
    lines = to_lines(PAGE, FR)
    seg = segment(lines, DESCRIPTION, FR)
    chunks = chunk_paragraphs(to_paragraphs(lines[seg.start : seg.end], FR), FR)
    assert [c.kind for c in chunks] == ["preamble", "articles"]
    assert chunks[1].articles == ["1", "2"]
    assert "travaux" in chunks[1].text  # "tra-" + "vaux" re-joined
    assert article_label("art. 12. — le ministre", FR) == "12"
    assert article_label("article 5 de la loi n° 90-11", FR) is None


def test_two_columns_are_read_one_after_the_other():
    left = [(50 + 40 * i, 100 + 12 * r, 85 + 40 * i, 110 + 12 * r, f"L{r}{i}") for r in range(10) for i in range(5)]
    right = [(320 + 40 * i, 100 + 12 * r, 355 + 40 * i, 110 + 12 * r, f"R{r}{i}") for r in range(10) for i in range(5)]
    text = layout_text(left + right, width=600, rtl=False)
    assert text.index("L94") < text.index("R00")
    rtl = layout_text(left + right, width=600, rtl=True)
    assert rtl.index("R90") < rtl.index("L04")
    assert rtl.splitlines()[0].split()[0] == "R04"  # right-to-left inside a line


def test_short_title_needs_a_header_and_section_heading_is_skipped():
    page = (
        "La loi relative à la santé sera publiée.\n"  # the title words in running text: not a start
        "L O I S Loi n° 18-11 du 18 Chaoual 1439 correspondant au 2 juillet 2018\n"
        "relative à la santé.\n————\nLe Président de la République,\nVu la Constitution ;\n"
        "Article 1er. — La présente loi a pour objet de fixer les dispositions et principes fondamentaux.\n"
    )
    lines = to_lines(page, FR)
    seg = segment(lines, "Relative à la santé.", FR)
    assert seg.title_found and lines[seg.start].text.startswith("L O I S Loi n° 18-11")
    quoted = to_lines("Vu la loi relative à la santé ;\nArticle 1er. — Texte.", FR)
    assert not segment(quoted, "Relative à la santé.", FR).title_found


def test_search_result_parsing():
    page = (
        '<tr bgcolor="#78a7b9"><td><a href="#075309" title="Texte N°075309">&nbsp;</a>[*]</td><td colspan=5 id=gras '
        "align=\"right\"><a href='javascript:JoOpen(\"2005\",\"010\",\"4\",\"F\")' title=\"Afficher le journal\">"
        "[D&eacutetail]</a></td></tr><tr><td colspan=6>Décret exécutif n° 05-67 du 30 Janvier 2005</td></tr>"
        "<tr><td colspan=6>MINISTERE DU COMMERCE</td></tr><tr><td colspan=6>JO N° 10 du 06 F&eacute;vrier 2005, "
        'Page 4</td></tr><tr><td colspan=6><font color="#808080">Portant création du comité.</font></td></tr>'
    )
    (row,) = parse_page(page)
    assert (row["id_text"], row["type_text"], row["jt_number"]) == ("075309", "Décret exécutif", "05-67")
    assert (row["oj_year"], row["oj_number"], row["official_journal_page"]) == (2005, 10, 4)
    assert str(row["signature_date"].date()) == "2005-01-30" and str(row["publication_date"].date()) == "2005-02-06"
    assert row["source"] == "MINISTERE DU COMMERCE" and row["description"] == "Portant création du comité."


def test_metadata_helpers():
    assert citation_fr("Loi n° 84-11 du 09 Juin 1984", 24, 1984, 612) == (
        "Loi n° 84-11 du 09 Juin 1984, Journal officiel n° 24 de 1984, p. 612"
    )
    assert is_administrative_fr("Portant nomination du directeur général des impôts.")
    assert not is_administrative_fr("Relative à la promotion de l'investissement.")
    assert not is_administrative("يتعلق بترقية الاستثمار.")
    assert is_administrative("يتضمن تعيين مدير الدراسات.")
