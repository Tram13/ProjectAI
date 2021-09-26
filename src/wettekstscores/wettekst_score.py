import os


def calculate_score(file_content_1, file_content_2):
    wet_teksten_1_raw = file_content_1.split('\n')[3]
    wet_teksten_2_raw = file_content_2.split('\n')[3]

    wet_teksten_1 = parse_wettekst(wet_teksten_1_raw)
    wet_teksten_2 = parse_wettekst(wet_teksten_2_raw)

    # print("wetteksten 1:")
    # for wettekst in wet_teksten_1:
    #     wettekst.print()
    # print("\nwetteksten 2:")
    # for wettekst in wet_teksten_2:
    #     wettekst.print()

    score_1, score_2 = score_algoritme(wet_teksten_1, wet_teksten_2)

    return score_1, score_2


def parse_wettekst(raw_wettekst):
    output = []
    splitted = raw_wettekst.split('||')
    splitted = [text.replace(" ", "") for text in splitted]
    splitted = [text.replace("\n", "") for text in splitted]
    splitted = [text for text in splitted if text != ""]
    for tekst in splitted:
        s = tekst.split("-")
        if len(s) == 1:
            output.append(Wettekst(s[0], None))
        else:
            output.append(Wettekst(s[0], s[1]))
    return output


def score_algoritme(wet_teksten_1, wet_teksten_2):
    if len(wet_teksten_1) == 0 or len(wet_teksten_2) == 0:
        return 0.05, 0.05

    totaal_2 = 0.0
    score_a_in_b = 0.0
    for wet_a in wet_teksten_1:
        tussen_score = 0.0
        for wet_b in wet_teksten_2:
            tussen_score = max(tussen_score, wet_a.equal_score(wet_b))
        score_a_in_b += tussen_score
    totaal_2 += score_a_in_b
    score_a_in_b /= len(wet_teksten_1)

    score_b_in_a = 0.0
    for wet_b in wet_teksten_2:
        tussen_score = 0.0
        for wet_a in wet_teksten_1:
            tussen_score = max(tussen_score, wet_b.equal_score(wet_a))
        score_b_in_a += tussen_score
    totaal_2 += score_b_in_a
    score_b_in_a /= len(wet_teksten_2)

    totaal_1 = score_a_in_b * 0.6 + score_b_in_a * 0.4
    totaal_2 = totaal_2 / (len(wet_teksten_1) + len(wet_teksten_2))
    return totaal_1, totaal_2


class Wettekst:
    def __init__(self, naam, artikel):
        self.naam = naam
        self.artikel = artikel

    def equal_score(self, wet_tekst_2):
        if self.naam == wet_tekst_2.naam:
            if self.artikel == wet_tekst_2.artikel:
                return 1
            else:
                return 0.7
        return 0

    def print(self):
        if self.artikel is not None:
            print("naam: {}, artikel: {}".format(self.naam, str(self.artikel)))
        else:
            print("naam: {}".format(self.naam))


if __name__ == '__main__':
    f1 = "ECLI_NL_CBB_2005_AS7074.txt"
    f2 = "ECLI_NL_CBB_2005_AS7076.txt"
    path_1 = os.path.join("data", "parsed_data", "2005", "200502", f1)
    path_2 = os.path.join("data", "parsed_data", "2005", "200502", f2)

    with open(path_1, 'r', encoding='utf-8') as f:
        f_i_1 = f.read()
    with open(path_2, 'r', encoding='utf-8') as f:
        f_i_2 = f.read()

    points_1, points_2 = calculate_score(f_i_1, f_i_2)
    print("File 1: {}, File 2: {}".format(f1, f2))
    print("Score 1: {}\nScore 2: {}".format(str(points_1), str(points_2)))
