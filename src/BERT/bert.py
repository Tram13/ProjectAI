from sentence_transformers import SentenceTransformer, util
import os

from src.BERT.score_bert import add_score, sort_scores, print_scores, empty_scores
from src.sentence_splitter import split_into_sentences
from thesis_master.src.load import load_docs
from thesis_master.src.models_implementation.preproces_pipeline import subject_and_case

model = SentenceTransformer("xlm-r-100langs-bert-base-nli-stsb-mean-tokens")


def _paraphrase_mining(sentence, text2_list):
    paraphrases = util.paraphrase_mining(model, [sentence] + text2_list, show_progress_bar=True)
    # Get highest score for current sentence, compared to text2
    max_score = 0
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        if i == 0 and j != 0:
            max_score = max(score, max_score)
    return max_score


def bert_texts_paraphrase_mining(text_list, compare_to_list):
    total_score = 0
    # paraphrase mining for each sentence from sentences1
    for sentence in text_list:
        total_score += _paraphrase_mining(sentence, compare_to_list)

    return total_score / len(text_list)  # average score


def bert_texts_as_sentences(sentence1, sentence2):
    query_embedding = model.encode(sentence1, convert_to_tensor=True)
    passage_embedding = model.encode(sentence2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embedding, passage_embedding).numpy()[0][0]
    return similarity


def get_index_of_source(source, docname_list):
    index = docname_list.index(source)
    return index


def get_filtered_lists(source, doc_list, docname_list):
    index = get_index_of_source(source, docname_list)
    filtered_list = doc_list.copy()
    filtered_namelist = docname_list.copy()
    del filtered_list[index]
    del filtered_namelist[index]
    return filtered_list, filtered_namelist


def do_bert_for_given_doc(source, doc_list, docname_list):
    source_doc = doc_list[get_index_of_source(source, docname_list)]
    filtered_list, filtered_namelist = get_filtered_lists(source, doc_list, docname_list)
    output = []
    for i, doc in enumerate(filtered_list):
        print("Working on file {} of {}.".format(i+1, len(filtered_list)))
        score = bert_texts_paraphrase_mining(source_doc, doc)
        add_score(source, filtered_namelist[i], score)
        # print("{} -> {}: {}".format(source, filtered_namelist[i], score))
        output.append((filtered_namelist[i], score))

    sort_scores()
    print_scores()
    return output


def do_bert(source, score_dict, INDEX_WETTEKSTEN, all=False, amount=4007):
    empty_scores()
    parseddata_path = os.path.join("data", "parsed_data", "2005")
    doc_list, docname_list, month_list = load_docs(parseddata_path)
    todo_docs = []
    todo_docnames = []
    index = 0
    for doc, name, month in zip(doc_list, docname_list, month_list):
        value = score_dict.get(name, None)
        if all or (value and value[INDEX_WETTEKSTEN] >= 0.01):
            todo_docs.append(doc)
            todo_docnames.append(name)
        if all:
            index += 1
        if index >= amount:
            break

    todo_docs = [split_into_sentences(subject_and_case(doc)[1]) for doc in todo_docs]
    # only do bert if wettekstenscore >= 0.01
    bert_scores = do_bert_for_given_doc(source.split(os.sep)[-1], todo_docs, todo_docnames)
    return bert_scores


# voorbeeld bert_texts_paraphrase_mining


# voorbeeld bert_texts_as_sentences
# t1 = 'Dit beroep slaagt niet.'
# t2 = 'Dit beroep slaagt wel.'
# print("Similarity: ", bert_texts_as_sentences(t1, t2)
# print(bert_texts_as_sentences(" ".join(roodkapje1), " ".join(roodkapje2)))
# print(bert_texts_as_sentences(" ".join(roodkapje1), " ".join(zeven_geitjes)))
