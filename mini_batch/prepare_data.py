import gzip
import cPickle
import numpy

def arg_passing(argv):
    # -data: dataset
    # -saving: log & model saving file
    # -dim: dimension of highway layers
    i = 1
    arg_dict = {'-data': 'pubmed',
                '-nlayers': 10,
                '-saving': 'pubmed',
                '-dim': 50,
                '-shared': 1,
                '-nmean': 1,
                '-reg': '',
                '-model': '',
                '-seed': 1234,
                '-y': 1,
                '-opt': 'RMS', # or Adam
                '-batch': 100}

    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i+1]
        i += 2

    arg_dict['-nlayers'] = int(arg_dict['-nlayers'])
    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-shared'] = int(arg_dict['-shared'])
    arg_dict['-nmean'] = int(arg_dict['-nmean'])
    arg_dict['-stSize'] = int(arg_dict['-stSize'])
    arg_dict['-seed'] = int(arg_dict['-seed'])
    arg_dict['-y'] = int(arg_dict['-y'])
    return arg_dict

def create_mask(rel_list):
    n_nodes = len(rel_list)
    n_rels = len(rel_list[0])
    max_neigh = 0

    for node in rel_list:
        for rel in node:
            max_neigh = max(max_neigh, len(rel))

    rel = numpy.zeros((n_nodes, n_rels, max_neigh), dtype='int64')
    mask = numpy.zeros((n_nodes,n_rels, max_neigh), dtype='float32')

    for i, node in enumerate(rel_list):
        for j, r in enumerate(node):
            n = len(r)
            rel[i, j, : n] = r
            mask[i, j, : n] = 1.0

    return rel, mask

def load_data(path):
    f = gzip.open(path, 'rb')
    feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)

    rel_list, rel_mask = create_mask(rel_list)

    return feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids

class MiniBatchIds():
    def __init__(self, n_samples, batch_size=100):
        self.ids = numpy.arange(n_samples)
        self.batch_size = batch_size

    def get_mini_batch_ids(self, batch_id):
        if batch_id == 0:
            numpy.random.shuffle(self.ids)

        return self.ids[self.batch_size * batch_id : self.batch_size * (batch_id + 1)]