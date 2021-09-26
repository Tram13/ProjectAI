from datetime import datetime
import os
import numpy as np
from gensim.models import Word2Vec

from thesis_master.src.models_implementation.model import Model
from thesis_master.src.models_implementation.preproces_pipeline import preprocess, subject_and_case
from thesis_master.src.load import load_docs

import pickle

GENERATE_WORD2VEC = True


class Word2VecModel(Model):

    def train(self):
        # Load cases and collecting cases
        if GENERATE_WORD2VEC:
            documents = load_docs(self.documents)

            subjects, cases = zip(*(subject_and_case(c) for c in documents[0]))

            # Preprocessing datasets
            cases = list(map(lambda c: preprocess(c), cases))

            data = (documents[1], subjects, cases)

            datadump = open(self.filename, 'wb')  # De gefilterde data opslaan
            pickle.dump(data, datadump)
            datadump.close()

        datadump = open(self.filename, 'rb')  # De gefilterde data openen
        data = pickle.load(datadump)
        datadump.close()

        model = Word2Vec(data[2], size=200, window=5, min_count=1, workers=4)  # Model maken

        print('All preprocessing done, training embedding')

        model.save(os.path.join(self.save_path, 'word2vec_model'))  # Model opslaan
        return model

    def query(self, case, modelname):
        # load cases
        casedump = open(self.filename, 'rb')
        cases = pickle.load(casedump)
        casedump.close()

        # loading model
        model = Word2Vec.load(os.path.join(self.save_path, modelname))

        # load documents
        documents = load_docs(self.documents)

        # prepare queried case
        case = preprocess(case)
        case_vec = vectorize(model, case)

        results = []
        for i, v in enumerate(cases[2]):
            sim_score = _cosine_sim(vectorize(model, v), case_vec)
            if sim_score > 0:
                results.append({"score": sim_score, "doc": i})

        results.sort(key=lambda k: k["score"], reverse=True)

        return documents, results

    def vectorspace_and_data(self, modelname):
        # get model
        model = Word2Vec.load(os.path.join(self.save_path, modelname))

        # get data
        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()

        # get the vectorspaces
        docvecs = [vectorize(model, case) for case in data[2]]

        return docvecs, data


def vectorize(model, doc):
    if len(doc) == 0:
        return np.full(200, 0, dtype=float)
    return np.mean([model.wv[word] for word in doc], axis=0)


def _cosine_sim(vec_a, vec_b):
    """Find the cosine similarity distance between two vectors."""
    csim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if np.isnan(np.sum(csim)):
        return 0
    return csim


def do_word2vec(generate_data=False, month="01", year=False, generate_scores=False):
    root = os.path.join("models", "word2vec", "2005")
    if not os.path.exists(root):
        os.makedirs(root)
    if not year:
        monthdir = str(2005) + str(month).zfill(2)
        monthdir_path = os.path.join("models", "word2vec", "2005", monthdir)
        if not os.path.exists(monthdir_path):
            os.mkdir(monthdir_path)
        preprocessed_data_path = os.path.join(monthdir_path, "word2vec_cases{}_preprocessed.data".format(monthdir))
        parseddata_path = os.path.join("data", "parsed_data", "2005", monthdir)
        w2v = Word2VecModel(
            preprocessed_data_path,     # filename
            monthdir_path,              # savepath
            parseddata_path             # documents
        )
        if generate_data:
            w2v.train()  # voor zover ik begrijp, genereert dit de effectieve word2vecs...
        # set path for scores
        if generate_scores:
            for dimension in [2, 5, 20, 100]:
                score_path = os.path.join(monthdir_path, "word2vec_scores{}_dimensions{}".format(monthdir, dimension))
                # calculate and save scores
                w2v.full_run("word2vec_model", reduced=True, dimensions=dimension, filepath=score_path)
            # De masterstudent voerde de laatste test ook uit met dimensions=100
            print(w2v.cluster_analysis("word2vec_model"))
    else:
        year_path = "year"
        year_dir = os.path.join("models", "word2vec", "2005", year_path)
        if not os.path.exists(year_dir):
            os.mkdir(year_dir)
        preprocessed_data_path = os.path.join(year_dir, "word2vec_cases_preprocessed.data")
        parseddata_path = os.path.join("data", "parsed_data", "2005")
        w2v = Word2VecModel(
            preprocessed_data_path,  # filename
            year_dir,  # savepath
            parseddata_path  # documents
        )
        if generate_data:
            w2v.train()  # voor zover ik begrijp, genereert dit de effectieve word2vecs...
        # set path for scores
        if generate_scores:
            for dimension in [2, 5, 20, 100]:
                score_path = os.path.join(year_dir, "word2vec_scores_dimensions{}".format(dimension))
                # calculate and save scores
                w2v.full_run("word2vec_model", reduced=True, dimensions=dimension, filepath=score_path)
            # De masterstudent voerde de laatste test ook uit met dimensions=100
            print(w2v.cluster_analysis("word2vec_model"))
    return w2v
