import os

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from thesis_master.src.models_implementation.model import Model
from thesis_master.src.models_implementation.preproces_pipeline import preprocess, subject_and_case
from thesis_master.src.load import load_docs

import pickle

GENERATE_DOC2VEC = True


class Doc2VecModel(Model):

    def train(self):

        if GENERATE_DOC2VEC:
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

        print('All preprocessing done, training embedding')
        cases = [TaggedDocument(doc, [i]) for i, doc in enumerate(data[2])]
        model = Doc2Vec(dm=0, alpha=0.025, vector_size=100, window=5, epochs=20, workers=8)
        model.build_vocab(cases, keep_raw_vocab=True)
        model.train(cases, epochs=model.epochs, total_examples=model.corpus_count)
        model.save(os.path.join(self.save_path, 'doc2vec_model'))  # Model opslaan

    def query(self, case, modelname):
        # loading model
        model = Doc2Vec.load(os.path.join(self.save_path, modelname))

        # load documents
        documents = load_docs(self.documents)

        # case
        case = preprocess(case)
        vec = model.infer_vector(case)

        # use the most_similar utility to find the most similar documents.
        similars = model.docvecs.most_similar(positive=[vec], topn=4007)

        return documents, similars

    def vectorspace_and_data(self, modelname):
        # get model
        model = Doc2Vec.load(os.path.join(self.save_path, modelname))

        # get data
        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()

        # get the vectorspaces
        docvecs = model.docvecs.vectors_docs

        return docvecs, data


def do_doc2vec(generate_data=False, year=True, generate_scores=False):
    root = os.path.join("models", "doc2vec", "2005")
    if not os.path.exists(root):
        os.makedirs(root)
    d2v = None
    if not year:
        for month in range(1, 13):
            monthdir = str(2005) + str(month).zfill(2)
            monthdir_path = os.path.join("models", "doc2vec", "2005", monthdir)
            if not os.path.exists(monthdir_path):
                os.mkdir(monthdir_path)
            preprocessed_data_path = os.path.join(monthdir_path, "doc2vec_cases{}_preprocessed.data".format(monthdir))
            parseddata_path = os.path.join("data", "parsed_data", "2005", monthdir)
            d2v = Doc2VecModel(
                preprocessed_data_path,     # filename
                monthdir_path,              # savepath
                parseddata_path             # documents
            )
            if generate_data:
                d2v.train()  # voor zover ik begrijp, genereert dit de effectieve word2vecs...
            # set path for scores
            if generate_scores:
                score_path = os.path.join(monthdir_path, "doc2vec_scores{}".format(monthdir))
                # calculate and save scores
                d2v.full_run("doc2vec_model", reduced=True, dimensions=2, filepath=score_path)  # Dit voert alle testen uit
                # De masterstudent voerde de laatste test ook uit met dimensions=100
                print(d2v.cluster_analysis("doc2vec_model"))
    else:
        year_path = "year"
        year_dir = os.path.join(root, year_path)
        if not os.path.exists(year_dir):
            os.mkdir(year_dir)
        preprocessed_data_path = os.path.join(year_dir, "doc2vec_cases_preprocessed.data")
        parseddata_path = os.path.join("data", "parsed_data", "2005")
        d2v = Doc2VecModel(
            preprocessed_data_path,  # filename
            year_dir,  # savepath
            parseddata_path  # documents
        )
        if generate_data:
            d2v.train()
        if generate_scores:
            score_path = os.path.join(year_dir, "doc2vec_scores")
            # calculate and save scores
            d2v.full_run("doc2vec_model", reduced=True, dimensions=2, filepath=score_path)  # Dit voert alle testen uit
            # De masterstudent voerde de laatste test ook uit met dimensions=100
            print(d2v.cluster_analysis("doc2vec_model"))
    return d2v
