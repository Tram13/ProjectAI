import xml.etree.ElementTree
import re
import os

# find . -name '*.txt' -exec head -n 1 {} \; | sort | uniq -c
path = os.path.join("data", "original", "2005")
parsed_data_path = os.path.join("data", "parsed_data", "2005")


class XMLParser:
    __counter_parsed = 0
    __counter_not_parsed = 0
    __reference_text = ''
    __inhoud_text = []
    __uitspraak_text = []

    def write_to_file(self, xmlfile, subject):
        filename = xmlfile.split(os.sep)[-1].split('.')[0]
        month = xmlfile.split(os.sep)[3]
        new_file = os.path.join(parsed_data_path, month, filename + ".txt")
        with open(new_file, 'w+', encoding="utf-8") as processed_file:
            processed_file.write(subject.text)
            processed_file.write('\n')
            processed_file.write(' '.join(
                [inh.text for inh in self.__inhoud_text if inh.text is not None and inh is not None and inh != '\n']))
            processed_file.write('\n')
            processed_file.write(
                ' '.join(
                    [uitspr.text for uitspr in self.__uitspraak_text if
                     uitspr is not None and uitspr.text is not None]))
            processed_file.write('\n')
            processed_file.write(self.__reference_text)
            processed_file.close()

    def is_valid_file(self, subject, inhoud, uitspraak, references):
        # if subject is not None and inhoud is not None and inhoud and  uitspraak is not None and
        # len(uitspraak_text) > 0:
        return subject is not None and inhoud is not None and inhoud and uitspraak is not None and len(
            self.__uitspraak_text) > 0 and len(references) > 0

    def get_uitspraak(self, uitspraak, namespaces):
        if uitspraak:
            for block in uitspraak:
                if len(block.findall('recht:para', namespaces)) > 0:
                    self.__uitspraak_text += block.findall('recht:para', namespaces)
                else:
                    self.__uitspraak_text.append(block)

    def get_inhoud(self, inhoud, namespaces):
        if inhoud:
            for block in inhoud:
                self.__inhoud_text += (block.findall('recht:para', namespaces))

    def get_references(self, references):
        if len(references) > 0:
            for ref in references:
                self.__reference_text += ''.join(
                    [c.lower() for c in re.sub('\(.*\)', '', ref.text) if c.isalpha() or c == ' ']).strip()
                if ref.attrib['{bwb-dl}resourceIdentifier']:
                    match = re.findall('artikel=([0-9]*)', ref.attrib['{bwb-dl}resourceIdentifier'])
                    if len(match) > 0 and match[0] != '':
                        self.__reference_text += ' - ' + match[0]
                self.__reference_text += ' || '

    def parseXML(self, xmlfile):
        # Resetting variables
        self.__reference_text = ''
        self.__inhoud_text = []
        self.__uitspraak_text = []

        namespaces = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'dcterms': "http://purl.org/dc/terms/",
            'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            'recht': 'http://www.rechtspraak.nl/schema/rechtspraak-1.0'
        }
        # create element tree object
        tree = xml.etree.ElementTree.parse(xmlfile)

        # get root element
        root = tree.getroot()

        rdf_metadata = root.find('rdf:RDF', namespaces)
        description = rdf_metadata.find('rdf:Description', namespaces)
        subject = description.find('dcterms:subject', namespaces)

        references = description.findall('dcterms:references', namespaces)
        self.get_references(references)

        inhoud = root.find('recht:inhoudsindicatie', namespaces)
        self.get_inhoud(inhoud, namespaces)

        uitspraak = root.find('recht:uitspraak', namespaces)
        self.get_uitspraak(uitspraak, namespaces)

        if self.is_valid_file(subject, inhoud, uitspraak, references):
            self.write_to_file(xmlfile, subject)
            self.__counter_parsed += 1
        else:
            self.__counter_not_parsed += 1

    def parse_all_data(self):
        self.__counter_parsed = 0
        # d = directories
        for _, d, _ in os.walk(path):
            for directory in d:
                for month_dir, _, files_in_dir in os.walk(os.path.join(path, directory)):
                    print("Working on directory {}...".format(month_dir))
                    target_dir = os.path.join(parsed_data_path, directory)
                    if not os.path.exists(target_dir):
                        os.mkdir(target_dir)
                    for file in files_in_dir:
                        if '.xml' in file:
                            self.parseXML(os.path.join(month_dir, file))

        print("{} texts with references found".format(self.__counter_parsed))
        print("{} texts could not be parsed".format(self.__counter_not_parsed))
