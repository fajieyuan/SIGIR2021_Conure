import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter


class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        index_data_file = options['dir_name_index']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        max_document_length = max([len(x.split(",")) for x in positive_examples])
        #max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping

        #combine dict
        # self.item_dict=dict(self.item_dict.items() + config.sym_dict.items())
        self.embed_len=len(self.item_dict)
        # write index map to the below file
        f = open(index_data_file, 'w')
        # f.write('Hello, world!')
        f.write(str(self.item_dict))
        f.close()
        print ("the index has been written to {}").format(index_data_file)








