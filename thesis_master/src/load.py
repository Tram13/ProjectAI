import os


def load_docs(folder_name):
    doc_list = []
    docname_list = []
    month_list = []
    for root, dirs, files in os.walk(folder_name):
        for directory in dirs:
            p = os.path.join(folder_name, directory)
            for name in os.listdir(p):
                if name.endswith('txt'):
                    docname_list.append(name)
                    st = open(os.path.join(p, name), 'r', encoding='utf-8').read()
                    doc_list.append(st)
                    month_list.append(directory)

    # doc_list = []
    # docname_list = [name for name in os.listdir(folder_name) if name.endswith('txt')]
    # file_list = [folder_name + '/' + name for name in docname_list if name.endswith('txt')]
    # for file in file_list:
    #     st = open(file, 'r', encoding='utf-8').read()
    #     doc_list.append(st)

    # print('Found %s documents under the dir %s' % (len(doc_list), folder_name))

    return doc_list, docname_list, month_list
