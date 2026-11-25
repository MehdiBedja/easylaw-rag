from easylaw_rag.process.chunk import article_label, chunk_paragraphs
from easylaw_rag.process.segment import segment
from easylaw_rag.text.clean import to_lines, to_paragraphs

PAGE = """الجريدة الرسمية للجمهورية الجزائرية / العدد 12
4
المادة 9 : ينشر هذا القرار في الجريدة الرسمية.
حرر بالجزائر في 3 مارس سنة 2021.
وزير المالية

مرسوم تنفيذي رقم 21-55 مؤرخ في 10 مارس سنة 2021
يحدد كيفيات تسيير حساب التخصيص الخاص للصندوق الوطني للسكن.

إن الوزير الأول،
- بناء على الدستور،
- وبمقتضى القانون رقم 84-17 المؤرخ في 7 يوليو سنة 1984،
يرسم ما يأتي :
المادة الأولى : يهدف هذا المرسوم إلى تحديد كيفيات تسيير
حساب التخصيص الخاص.
المادة 2 : يفتح في كتابات الخزينة حساب
تخصيص خاص.
حرر بالجزائر في 10 مارس سنة 2021.
عبد العزيز جراد

قرار مؤرخ في 12 مارس سنة 2021 يتضمن تعيين أعضاء اللجنة الوطنية.
"""
DESCRIPTION = "يحدد كيفيات تسيير حساب التخصيص الخاص للصندوق الوطني للسكن."


def test_noise_lines_removed():
    lines = to_lines(PAGE)
    assert lines[0] is None  # running header
    assert lines[1] is None  # page number


def test_segment_cuts_previous_and_next_text():
    lines = to_lines(PAGE)
    seg = segment(lines, DESCRIPTION)
    assert seg.title_found
    body = "\n".join(l.text for l in lines[seg.start : seg.end] if l)
    assert body.startswith("مرسوم تنفيذي رقم 21-55")
    assert "عبد العزيز جراد" in body
    assert "وزير المالية" not in body
    assert "اللجنة الوطنية" not in body


def test_multiline_visa_never_ends_the_text():
    # Regression: a visa wrapped as "... الموافق ... الذي" / "يحدد ..." looks like a title header.
    page = """قرار مؤرخ في 19 مارس سنة 2018 يحدد غابة الاستجمام سيدي سليمان شرق.
إن وزير الفلاحة،
- وبمقتضى المرسوم التنفيذي رقم 427-12 المؤرخ
في 2 صفر عام 1434 الموافق 16 ديسمبر سنة 2012 الذي
يحدد شروط وكيفيات إدارة وتسيير الأملاك العمومية والخاصة التابعة للدولة،
يقرر ما يأتي :
المادة الأولى : يحدد هذا القرار غابة الاستجمام سيدي سليمان شرق.
حرر بالجزائر في 19 مارس سنة 2018.
"""
    lines = to_lines(page)
    seg = segment(lines, "يحدد غابة الاستجمام سيدي سليمان شرق.")
    body = "\n".join(l.text for l in lines[seg.start : seg.end] if l)
    assert "المادة الأولى" in body
    assert "حرر بالجزائر" in body


def test_chunks_are_article_aware():
    lines = to_lines(PAGE)
    seg = segment(lines, DESCRIPTION)
    chunks = chunk_paragraphs(to_paragraphs(lines[seg.start : seg.end]))
    assert [c.kind for c in chunks] == ["preamble", "articles"]
    assert chunks[1].articles == ["1", "2"]
    assert "تخصيص خاص" in chunks[1].text


def test_article_label_tolerates_ocr():
    assert article_label("الماده 12 : يحدد") == "12"
    assert article_label("الملاه 3 يحدد") == "3"
    assert article_label("الماده الاولي : يهدف") == "1"
    assert article_label("الماده ؟ : ينشر") == "?"
    assert article_label("وبمقتضي القانون") is None


def test_citation_without_number_or_date():
    import pandas as pd

    from easylaw_rag.process.metadata import citation

    c = citation("قرار", float("nan"), pd.NaT, 27, 1979, 621)
    assert c == "قرار، الجريدة الرسمية عدد 27 لسنة 1979، ص 621"
    assert "nan" not in citation("قرار", None, pd.Timestamp("1979-06-06"), 27, 1979, 621)


def test_individual_acts_are_administrative():
    from easylaw_rag.process.metadata import is_administrative

    assert is_administrative("يتضمن التجنس بالجنسية الجزائرية.")
    assert is_administrative("يتضمن تفويض الإمضاء إلى مدير الموارد البشرية.")
    assert is_administrative("يتضمن تحويل اعتماد إلى ميزانية تسيير وزارة الثقافة.")
    assert not is_administrative("يتضمن قانون الجنسية الجزائرية.")
    assert not is_administrative("يتعلق بعلاقات العمل.")


def test_reversed_ocr_words_are_restored():
    from easylaw_rag.text.arabic import unreverse_words

    assert unreverse_words("ىضتقمبو رمالا خرؤملا ىف 11 يوليو") == "وبمقتضى الامر المؤرخ فى 11 يوليو"
    clean = "وبمقتضى الامر المؤرخ في 11 يوليو سنة 1970 والمتعلق بتنظيم سوق الحبوب"
    assert unreverse_words(clean) == clean
    assert unreverse_words("لا يجوز ذلك إلا بترخيص") == "لا يجوز ذلك إلا بترخيص"
