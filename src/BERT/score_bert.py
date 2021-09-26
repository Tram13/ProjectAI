import os
import pickle


class AllSores:
    def __init__(self):
        self.bert_scores = []

    def add(self, bert_score):
        self.bert_scores.append(bert_score)

    def sort_scores(self):
        self.bert_scores.sort(key=lambda x: -x.score)

    def print_data(self):
        if not self.bert_scores:
            print("No scores found")
        for bert_score in self.bert_scores:
            bert_score.print_data()


class BertScore:
    def __init__(self, file_name_1, file_name_2, score):
        self.file_name_1 = file_name_1
        self.file_name_2 = file_name_2
        self.score = score

    def print_data(self):
        print("File 1: {}, File 2: {} with similarity score of {}.".format(self.file_name_1, self.file_name_2, str(self.score)))


def get_file_path_bert():
    current_path = os.path.join("models", "BERT")
    if not os.path.exists(current_path):
        os.mkdir(current_path)
    current_path = os.path.join(current_path, "Score")
    if not os.path.exists(current_path):
        os.mkdir(current_path)
    current_path = os.path.join(current_path, "score.data")
    return current_path


def save_score(data):
    file_path = get_file_path_bert()
    datadump = open(file_path, 'wb')
    pickle.dump(data, datadump)
    datadump.close()


def read_score():
    file_path = get_file_path_bert()
    a_file = open(file_path, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output


def add_score(file_name_1, file_name_2, score):
    path = get_file_path_bert()
    # first time -> make empty
    if not os.path.exists(path):
        data = AllSores()
    else:
        data = read_score()
    data.add(BertScore(file_name_1, file_name_2, score))
    save_score(data)


def print_scores():
    path = get_file_path_bert()
    if not os.path.exists(path):
        data = AllSores()
    else:
        data = read_score()
    data.print_data()


def empty_scores():
    save_score(AllSores())


def sort_scores():
    path = get_file_path_bert()
    if os.path.exists(path):
        data = read_score()
        data.sort_scores()
        save_score(data)


if __name__ == '__main__':
    empty_scores()
    add_score("sfsdf", "sdfsdf", 0.6)
    add_score("abc", "def", 1)
    add_score("qqq", "aba", 0.5)
    add_score("sfsdf", "sdfsdf", 0.7)
    print_scores()
    print("---------")
    sort_scores()
    print_scores()

