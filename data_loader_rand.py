import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter
import config


class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        index_data_file = options['dir_name_index']
        self.item_dict = self.read_dict(index_data_file)

        positive_examples = list(open(positive_data_file, "r").readlines())
        # positive_examples = [s for s in positive_examples]

        colon = ",,"
        source = [s.split(colon)[0] for s in positive_examples]
        target = [s.split(colon)[1] for s in positive_examples]

        max_document_length = max([len(x.split(",")) for x in source])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = self.map_dict(self.item_dict, source)

        max_document_length_target = max([len(x.split(",")) for x in target])
        vocab_processor_target = learn.preprocessing.VocabularyProcessor(max_document_length_target)
        self.target = np.array(list(vocab_processor_target.fit_transform(target)))  # pad 0 in the end
        self.target_dict = vocab_processor_target.vocabulary_._mapping

        self.separator = 0  # denote '[CLS]'
        lens = self.item.shape[0]

        self.example = np.array([])
        self.example = []

        self.maxsource = len(self.item_dict)
        # GEN_Task = self.item_dict['GEN_Task4'] #1
        for line in range(lens):
            source_line = self.item[line]
            target_line = self.target[line]
            target_num = len(target_line)
            for j in range(target_num):
                if target_line[j] != 0:
                    # np.array(target_line[j])
                    # unit = np.append(np.array(self.separator),source_line)
                    # unit = np.append(GEN_Task, source_line) # 2
                    unit = np.append(source_line, np.array(self.separator)) #3
                    # unit = np.append(unit, np.array(target_line[j] + self.maxsource))
                    unit = np.append(unit, np.array(target_line[j] ))
                    self.example.append(unit)
        self.example = np.array(self.example)  # GEN_Task1 1 2 3 4...50 END/separator classifiere
        # self.embed_len=config.embed_len
        self.embed_len = len(self.item_dict)  # 4

    def read_dict(self, index_data_file):
        dict_temp = {}
        file = open(index_data_file, 'r')
        for line in file.readlines():
            dict_temp = eval(line)
        # print dict_temp
        return dict_temp

    def map_dict(self, dict_pretrain, source):
        items = []
        for lines in source:
            trueline = [dict_pretrain[x] for x in lines.split(',')]
            # print trueline
            trueline = np.array(trueline)
            items.append(trueline)
        return np.array(items)







