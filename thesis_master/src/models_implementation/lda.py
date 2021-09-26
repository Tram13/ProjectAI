import os

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from scipy.stats import entropy
from scipy import spatial

from thesis_master.src.models_implementation.model import Model
from thesis_master.src.models_implementation.preproces_pipeline import preprocess, subject_and_case
from thesis_master.src.load import load_docs

import pickle
import numpy as np

GENERATE_LDA = True


class LDAModel(Model):

    def __init__(self, filename, save_path, documents, dictname):
        self.dictname = dictname
        super(LDAModel, self).__init__(filename, save_path, documents)

    def train(self):
        if GENERATE_LDA:
            # Load cases and collecting cases
            documents = load_docs(self.documents)

            subjects, cases = zip(*(subject_and_case(c) for c in documents[0]))

            # Preprocessing datasets
            cases = list(map(lambda c: preprocess(c), cases))
            data = (documents[1], subjects, cases)

            datadump = open(self.filename, 'wb')
            pickle.dump(data, datadump)
            datadump.close()

        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()

        print('All preprocessing done, training model')

        cases_dict = Dictionary(data[2])
        cases_dict.filter_extremes(no_below=10, no_above=0.5)
        cases_dict.save_as_text(self.dictname)
        corpus = [cases_dict.doc2bow(text) for text in data[2]]

        print('Number of unique tokens: %d' % len(cases_dict))
        print('Number of documents: %d' % len(corpus))

        model = LdaModel(corpus=corpus, num_topics=80, id2word=cases_dict,
                         alpha=1e-2, eta=0.5e-2, chunksize=300, minimum_probability=0.0, passes=2)

        print(model.show_topics(num_topics=10, num_words=20))

        model.save(os.path.join(self.save_path, 'lda_model'))  # Model opslaan
        return model

    def query(self, case, modelname):
        # Load datamodel and dictionary
        model = LdaModel.load(os.path.join(self.save_path, modelname))

        # model = train()
        cases_dict = Dictionary.load_from_text(self.dictname)

        # Get Corpus
        casedump = open(self.filename, 'rb')
        data = pickle.load(casedump)
        casedump.close()
        corpus = [cases_dict.doc2bow(text) for text in data[2]]
        documents = load_docs(os.path.join("data", "parsed_data", "2005", "200501"))
        # documents = load_docs(self.documents)

        # Get vector of new case
        vec_bow = cases_dict.doc2bow(preprocess(case))
        new_doc_distribution = np.array(
            [tup[1] for tup in model.get_document_topics(bow=vec_bow, minimum_probability=0.0)])
        # Calculate similarity
        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in model[corpus]])
        best_score = [np.inf, np.inf, np.inf, np.inf, np.inf]
        for doc in doc_topic_dist:
            distance = 1 - spatial.distance.cosine(new_doc_distribution, doc)
            i = 0
            while i < 5 and distance > best_score[i]:
                i += 1
            if i != 5:
                best_score[i] = distance

        print(best_score)

    def vectorspace_and_data(self, modelname):
        # get model
        model = LdaModel.load(os.path.join(self.save_path, modelname))

        # get dictionary
        cases_dict = Dictionary.load_from_text(self.dictname)

        # get data
        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()
        corpus = [cases_dict.doc2bow(text) for text in data[2]]

        # get the vectorspaces
        docvecs = np.array([[tup[1] for tup in lst] for lst in model[corpus]])

        return docvecs, data


def jensen_shannon(query, matrix):
    p = query[None, :].T
    q = matrix.T
    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))


def get_most_similar_documents(query, matrix, k=10):
    sims = jensen_shannon(query, matrix)
    return sims.argsort()[:k], sorted(sims)[:k]


def do_lda(generate_data=False, year=True, generate_scores=False):
    root = os.path.join("models", "lda", "2005")
    if not os.path.exists(root):
        os.makedirs(root)
    lda = None
    if not year:
        for month in range(1, 13):
            monthdir = str(2005) + str(month).zfill(2)
            monthdir_path = os.path.join("models", "lda", "2005", monthdir)
            if not os.path.exists(monthdir_path):
                os.mkdir(monthdir_path)
            preprocessed_data_path = os.path.join(monthdir_path, "lda_cases{}_preprocessed.data".format(monthdir))
            parseddata_path = os.path.join("data", "parsed_data", "2005", monthdir)
            lda = LDAModel(
                preprocessed_data_path,  # filename
                monthdir_path,  # savepath
                parseddata_path,  # documents
                os.path.join("data", "original", "lda_2005_dict")  # Dict location
            )
            if generate_data:
                lda.train()  # voor zover ik begrijp, genereert dit de effectieve word2vecs...
            # set path for scores
            if generate_scores:
                for dimension in [2, 5, 20]:
                    score_path = os.path.join(monthdir_path, "lda_scores{}_dimensions{}".format(monthdir, dimension))
                    # calculate and save scores
                    lda.full_run("lda_model", reduced=True, dimensions=dimension, filepath=score_path)
                # De masterstudent voerde de laatste test ook uit met dimensions=20
                print(lda.cluster_analysis("lda_model"))
                lda.plot_vectorspace("tsne", "tsne", "lda_model")
    else:
        year_path = "year"
        year_dir = os.path.join("models", "lda", "2005", year_path)
        if not os.path.exists(year_dir):
            os.mkdir(year_dir)
        parseddata_path = os.path.join("data", "parsed_data", "2005")
        preprocessed_data_path = os.path.join(year_dir, "lda_cases_preprocessed.data")
        lda = LDAModel(
            preprocessed_data_path,  # filename
            year_dir,  # savepath
            parseddata_path,  # documents
            os.path.join("data", "original", "lda_2005_dict")  # Dict location
        )
        if generate_data:
            lda.train()  # voor zover ik begrijp, genereert dit de effectieve word2vecs...
        # set path for scores
        if generate_scores:
            for dimension in [2, 5, 20]:
                score_path = os.path.join(year_dir, "lda_scores_dimensions{}".format(dimension))
                # calculate and save scores
                lda.full_run("lda_model", reduced=True, dimensions=dimension, filepath=score_path)
            # De masterstudent voerde de laatste test ook uit met dimensions=20
            print(lda.cluster_analysis("lda_model"))
            lda.plot_vectorspace("tsne", "tsne", "lda_model")

        # text_file = "ECLI_NL_CBB_2005_AT3914.txt"
        # f = os.path.join("data", "parsed_data", "2005", "200503", text_file)
        # doc = open(f, 'r', encoding='utf-8').read()
        # _, case_ = subject_and_case(doc)
        # lda.query(case_, "lda_model")
    return lda
