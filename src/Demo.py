# OPMERKING: workdirectory zetten op de ROOT van het project!
import os

from BERT.bert import bert_texts_paraphrase_mining
from thesis_master.src.models_implementation.doc2vec import do_doc2vec
from thesis_master.src.models_implementation.preproces_pipeline import subject_and_case
from thesis_master.src.models_implementation.word2vec import do_word2vec


def get_case_from_document_path(filepath):  # code-duplicatie, what you gonna do
    doc = open(filepath, 'r', encoding='utf-8').read()
    _, case_ = subject_and_case(doc)
    return case_


# Roodkapje - 7 geiten test - BERT
if input("do BERT?: ") == "ja":
    roodkapje1 = [
        "Op een keer gaf ze haar een kapje van rood fluweel en omdat het haar zo lief stond en ze nooit meer iets anders "
        "wilde dragen, werd ze voortaan alleen maar Roodkapje genoemd.",
        "Op een dag zei haar moeder: \"Kijk eens, Roodkapje, daar heb je een stuk taart en een fles wijn, breng dat maar "
        "naar grootmoeder.\"",
        "\"Ze is ziek en sukkelt en het zal haar goed doen.\"",
        "\"Ga naar haar toe voor het te warm wordt en blijf op de grote weg, anders val je en dan breekt de fles en heeft"
        " onze arme oma er niets aan.\"",
        "\"En als je bij haar kamer komt, vergeet dan niet haar goedemorgen te wensen.\""
        "\"Ik zal aan alles denken, moeder,\" zei Roodkapje en gaf er haar de hand op.",
        "Maar grootmoeder woonde in het bos, wel een half uur buiten het dorp.",
        "Toen Roodkapje in het bos kwam, kwam ze de wolf tegen.",
        "Roodkapje wist niet dat het een boosaardig beest was en was helemaal niet bang voor hem.",
        "\"Dag, Roodkapje\", zei hij.",
        "\"Dag, Wolf\", zei Roodkapje."
    ]

    roodkapje2 = [
        "Er was eens een meisje dat Roodkapje heette, omdat ze altijd een mooi rood mutsje op haar hoofd droeg.",
        "Op een dag riep haar moeder Roodkapje bij zich.",
        "Ze gaf haar een mandje met een fles wijn, koekjes, appels en een heerlijke peperkoek.",
        "\"Grootmoeder is een beetje ziek\", zei ze.",
        "\"Wil jij dit mandje naar haar toebrengen?\"",
        "Natúúrlijk wilde roodkapje dat doen.",
        "\"Maar wel op het pad blijven, hoor!\","
        "\"Ja mamma, dat beloof ik.\"",
        "En daar ging Roodkapje, op weg naar haar grootmoeder.",
        "Onderweg kwam Roodkapje een wolf tegen.",
        "\"Dag meisje\", zei Wolf.",
        "\"Dag, Wolf\", zei Roodkapje."
    ]

    zeven_geitjes = [
        "Er was eens een oude geit die zeven jonge geitjes had en zij had ze lief zoals een moeder haar kinderen "
        "liefheeft.",
        "Op een dag wilde zij het bos ingaan om voedsel te halen: zij riep ze alle zeven bij elkaar en zei:",
        "\"Lieve kinderen, ik ga naar het bos, wees op je hoede voor de wolf.\"",
        "\"Als hij binnen komt, dan eet hij jullie allen met huid en haar op.\"",
        "\"De booswicht vermomt zich vaak, maar aan zijn rauwe stem en zijn zwarte poten kunnen jullie hem meteen "
        "herkennen.\"",
        "De geitjes zeiden: \"Lieve moeder, wij zullen goed oppassen, u kunt rustig weggaan.\"",
        "Toen mekkerde de oude geit en ging met een gerust hart op pad.",
        "Het duurde niet lang of er klopte iemand aan de voordeur die riep: \"Doe open, lieve kinderen.\"",
        "\"Ik ben het, moeder, ik heb voor jullie allemaal iets meegebracht.\"",
        "Maar de geitjes hoorden aan de rauwe stem dat het de wolf was.",
        "\"Wij doen niet open\", riepen zij.",
        "\"jij bent onze moeder niet, die heeft een zachte liefelijke stem, maar jouw stem is rauw; jij bent de wolf!\""
    ]
    score1 = bert_texts_paraphrase_mining(roodkapje1, roodkapje2)
    score2 = bert_texts_paraphrase_mining(roodkapje1, zeven_geitjes)
    score3 = bert_texts_paraphrase_mining(roodkapje2, zeven_geitjes)
    score4 = bert_texts_paraphrase_mining(roodkapje2, roodkapje1)
    print()
    print("Score roodkapje1 - roodkapje2: " + str(score1))
    print("Score roodkapje1 - 7geitjes: " + str(score2))
    print("Score roodkapje2 - 7geitjes: " + str(score3))
    print("Score roodkapje2 - roodkapje1: " + str(score4))

if input("do doc2vec?: ") == "ja":
    # Test van Doc2Vec
    doc_to_vec_model = do_doc2vec(generate_data=False, year=True, generate_scores=False)
    f = os.path.join("data", "parsed_data", "2005", "200501", "ECLI_NL_CBB_2005_AS2031.txt")
    case = get_case_from_document_path(f)
    docs, results = doc_to_vec_model.query(case, "doc2vec_model")
    result1 = results[0]
    result2 = results[1]
    print("DOC2VEC: match: {}  (similarity {})".format(docs[1][result1[0]], result1[1]))
    print("DOC2VEC: match: {}  (similarity {})".format(docs[1][result2[0]], result2[1]))

if input("do word2vec?: ") == "ja":
    word_to_vec_model = do_word2vec(generate_data=False, year=True, generate_scores=False)
    f = os.path.join("data", "parsed_data", "2005", "200501", "ECLI_NL_CBB_2005_AS2031.txt")
    case = get_case_from_document_path(f)
    docs, results = word_to_vec_model.query(case, "word2vec_model")
    result1 = results[0]
    result2 = results[1]
    print("WORD2VEC: match: {}  (similarity {})".format(docs[1][result1['doc']], result1['score']))
    print("WORD2VEC: match: {}  (similarity {})".format(docs[1][result2['doc']], result2['score']))

# Ten slotte bespreken we in de demo het resultaat als we een gemiddelde nemen van alle scores, zoals beschreven staat
# in het verslag
# Om het resultaat te verifiëren, kan men de main.py uitvoeren, met wkdir als de root van het project.

