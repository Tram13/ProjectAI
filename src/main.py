import os

from src.XMLParser.XMLParser import XMLParser
from src.BERT.bert import do_bert
from src.wettekstscores.wettekst_score import calculate_score
from thesis_master.src.models_implementation.lda import do_lda
from thesis_master.src.models_implementation.word2vec import do_word2vec
from thesis_master.src.models_implementation.doc2vec import do_doc2vec
from thesis_master.src.models_implementation.preproces_pipeline import subject_and_case
from thesis_master.src.load import load_docs

# Opmerking voor opstellen modellen: we gaan er van uit dat de gegeven case reeds verwerkt is in het model.
# Dit is een logische aannname, want om een case te vergelijken, moet er al een uitspraak geweest zijn, en kan die
# uitspraak ook toegevoegd worden aan het model. Dit kan zelfs iteratief gebeuren als er nieuwe cases bijkomen dmv
# model.train([["hello", "world"]], total_examples=1, epochs=1)


# To parse the dataset "2005", move the data to data/original/2005
PARSE_DATA = False
# To create Word2Vec models of 2005
WORD2VEC = True
# To create Doc2Vec models of 2005
DOC2VEC = True
# To create LDA models of 2005
LDA = False
# To create BERT models of 2005
BERT = False
# check similarity of wetteksten
WETTEKSTEN = True


# test voor gelijkaardige labels
TEST = True
# top n worden getest, -1 = allemaal
TEST_AMOUNT = -1


# if we want to print seperate results of w2v, ...
PRINT = True


def get_case_from_document_path(filepath):
    doc = open(filepath, 'r', encoding='utf-8').read()
    subjects, case_ = subject_and_case(doc)
    return case_, subjects


if PARSE_DATA:
    XMLParser().parse_all_data()

text_file = "ECLI_NL_CBB_2005_AS2031.txt"
monthdir = "200501"

#
# score is 0 als er geen overreenkomende wetteksten zijn (behalve als er een van de twee geen heeft)
# score dict: "naam_file": [score wetteksten, score w2v, score d2v, score BERT]
score_dict = {}

# indexes
INDEX_WETTEKSTEN = 0
INDEX_WORD2VEC = 1
INDEX_DOC2VEC = 2
INDEX_BERT = 3
WETTEKST_ALGORITME = 0  # use algoritm 0 or 1


# score formula
def score_formula(score_list):
    if score_list[INDEX_WETTEKSTEN] >= 0.01:
        return 0.3 * score_list[INDEX_WORD2VEC] + 0.3 * score_list[INDEX_DOC2VEC] \
               + 0.3 * score_list[INDEX_BERT] + 0.1 * score_list[WETTEKSTEN]
    return 0.0


# check labels
def check_labels(l_i, l_t):
    for label in l_t:
        if label in l_i:
            return True
    return False


# look for month_dir
def find_month(folder_name, f_name):
    for root, dirs, files in os.walk(folder_name):
        for directory in dirs:
            p = os.path.join(folder_name, directory)
            for name_f in os.listdir(p):
                if name_f == f_name:
                    return directory
    return None


# print top n in ranking, -1 = print all
PRINT_TOP = -1

if PRINT:
    print("\n\n\n START: WETTEKSTEN \n\n\n")

if WETTEKSTEN:
    print("Doing Wetteksten...")
    f = os.path.join("data", "parsed_data", "2005", monthdir, text_file)
    with open(f, 'r', encoding='utf-8') as f:
        f_1 = f.read()
    doc_path = os.path.join("data", "parsed_data", "2005")
    doc_contents, doc_names, _ = load_docs(doc_path)
    for content, name in zip(doc_contents, doc_names):
        score_1, score_2 = calculate_score(f_1, content)

        # add to score_dict
        key = name
        new_list = score_dict.get(key, [0, 0, 0, 0])
        new_list[INDEX_WETTEKSTEN] = score_1
        score_dict[key] = new_list

        if PRINT and score_1 > 0.01:
            print("WETTEKST: match: {}  (similarity {})".format(name, str(score_1)))

if PRINT:
    print("\n\n\n START: WORD2VEC \n\n\n")

if WORD2VEC:
    print("Doing Word2Vec...")
    word_to_vec_model = do_word2vec(generate_data=False, year=True, generate_scores=False)
    f = os.path.join("data", "parsed_data", "2005", monthdir, text_file)
    case, subjects = get_case_from_document_path(f)
    docs, results = word_to_vec_model.query(case, "word2vec_model")
    if TEST:
        labels_input = [s.strip() for s in subjects.split(';')]
        base_path = os.path.join("data", "parsed_data", "2005")
        for index, result in enumerate(results):
            if TEST_AMOUNT == -1 or index < TEST_AMOUNT:
                with open(os.path.join(base_path, docs[2][docs[1].index(docs[1][result['doc']])], docs[1][result['doc']]), 'r',
                          encoding='utf-8') as f:
                    st = f.read()
                    labels_test = [s.strip() for s in st.split("\n")[0].split(';')]
                    same_label = check_labels(labels_input, labels_test)

                print("WORD2VEC Same label? {}, match: {}  (similarity {})".format(same_label, docs[1][result['doc']],
                                                                                  result['score']))
            else:
                break
    elif PRINT:
        for result in results:
            print("WORD2VEC: match: {}  (similarity {})".format(docs[1][result['doc']], result['score']))

    # add to score_dict
    for result in results:
        key = docs[1][result['doc']]
        new_list = score_dict.get(key, [0, 0, 0, 0])
        new_list[INDEX_WORD2VEC] = result['score']
        score_dict[key] = new_list


if PRINT:
    print("\n\n\n START: DOC2VEC \n\n\n")


if DOC2VEC:
    print("Doing Doc2Vec...")
    doc_to_vec_model = do_doc2vec(generate_data=False, year=True, generate_scores=False)
    f = os.path.join("data", "parsed_data", "2005", monthdir, text_file)
    case, subjects = get_case_from_document_path(f)
    docs, results = doc_to_vec_model.query(case, "doc2vec_model")
    if TEST:
        labels_input = [s.strip() for s in subjects.split(';')]
        base_path = os.path.join("data", "parsed_data", "2005")
        for index, result in enumerate(results):
            if TEST_AMOUNT == -1 or index < TEST_AMOUNT:
                with open(os.path.join(base_path, docs[2][docs[1].index(docs[1][result[0]])], docs[1][result[0]]), 'r', encoding='utf-8') as f:
                    st = f.read()
                    labels_test = [s.strip() for s in st.split("\n")[0].split(';')]
                    same_label = check_labels(labels_input, labels_test)

                print("DOC2VEC Same label? {}, match: {}  (similarity {})".format(same_label, docs[1][result[0]], result[1]))
            else:
                break
    elif PRINT:
        for result in results:
            print("DOC2VEC: match: {}  (similarity {})".format(docs[1][result[0]], result[1]))

    # add results to score_dict
    for result in results:
        key = docs[1][result[0]]
        new_list = score_dict.get(key, [0, 0, 0, 0])
        new_list[INDEX_DOC2VEC] = result[1]
        score_dict[key] = new_list

if PRINT:
    print("\n\n\n START: LDA \n\n\n")

if LDA:
    print("Doing LDA...")
    lda_model = do_lda(generate_data=False, year=True, generate_scores=False)
    f = os.path.join("data", "parsed_data", "2005", monthdir, text_file)
    case, subjects = get_case_from_document_path(f)
    docs, ids, results = lda_model.query(case, "lda_model")
    if PRINT:
        for result in results:
            print("LDA: match: {}  (similarity {})".format(docs[1][ids[0]], 1 - result))

if PRINT:
    print("\n\n\n START: BERT \n\n\n")


if BERT and WETTEKSTEN:
    print("Doing BERT...")
    source = os.path.join("data", "parsed_data", "2005", monthdir, text_file)
    _, subjects = get_case_from_document_path(source)
    bert_scores = do_bert(source, score_dict, INDEX_WETTEKSTEN, all=TEST, amount=100)

    if TEST:
        labels_input = [s.strip() for s in subjects.split(';')]
        base_path = os.path.join("data", "parsed_data", "2005")
        for index, bert_score_object in enumerate(bert_scores):
            file_name = bert_score_object[0]
            bert_score = bert_score_object[1]
            if TEST_AMOUNT == -1 or index < TEST_AMOUNT:
                month = find_month(base_path, file_name)
                with open(os.path.join(base_path, month, file_name), 'r', encoding='utf-8') as f:
                    st = f.read()
                    labels_test = [s.strip() for s in st.split("\n")[0].split(';')]
                    same_label = check_labels(labels_input, labels_test)

                print("BERT Same label? {}, match: {}  (similarity {})".format(same_label, file_name, bert_score))
            else:
                break

    elif PRINT:
        for file_name, bert_score in bert_scores:
            print("BERT: match: {}  (similarity {})".format(file_name, str(bert_score)))

    # add to score_dict
    for file_name, bert_score in bert_scores:
        key = file_name
        new_list = score_dict.get(key, [0, 0, 0, 0])
        new_list[INDEX_BERT] = bert_score
        score_dict[key] = new_list

if not TEST:

    print("Calculating top score...")

    score_tuples = []
    for key, value in score_dict.items():
        score_tuples.append((key, score_formula(value)))

    # sort score_tuples
    score_tuples.sort(key=lambda x: -x[1])

    print("\n\nRANKING: \n")
    # print ranking:
    for index, score_tuple in enumerate(score_tuples):
        if index < PRINT_TOP or PRINT_TOP == -1:
            print("{}. Match {} with simularity score of {}".format(str(index + 1), score_tuple[0], str(score_tuple[1])))
        else:
            break


