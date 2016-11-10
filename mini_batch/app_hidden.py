import numpy
import keras.backend as K

class HiddenSaving():
    def __init__(self, n_samples, n_layers, h_dim, rel, rel_mask):
        self.curr_hidds = [numpy.zeros((n_samples, h_dim), dtype='float32') for t in range(n_layers)]
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.n_rel, self.n_neigh = rel.shape[1:]
        self.rel = rel              # (n_samples, n_rel, n_neigh)
        self.rel_mask = rel_mask    # (n_samples, n_rel, n_neigh)

    def get_context(self, list_ids):
        esp = 1e-8
        n_nodes = len(list_ids)
        mask = self.rel_mask[list_ids]

        contexts, masks = [], []
        for i in range(self.n_layers):
            context = self.curr_hidds[i][self.rel[list_ids].flatten()].reshape([n_nodes, self.n_rel, self.n_neigh, self.h_dim])
            # context = (n_nodes, n_rel, n_neigh, h_dim)
            # mask = (n_nodes, n_rel, n_neigh) -> sum n_neigh
            context = context * mask[:, :, :, None]
            context = numpy.sum(context, axis=-2) / (numpy.sum(mask, axis=-1) + esp)[:, :, None]
            contexts.append(context)

        return contexts

    def update_hidden(self, list_ids, hidds):
        for i in range(self.n_layers):
            self.curr_hidds[i][list_ids] = hidds[i]

def get_hidden_funcs(model, n_layers):
    hidd_layer = None
    dense_layer = None
    for layer in model.layers:
        if 'dense' in layer.name and dense_layer is None:
            dense_layer = layer.name

        if 'graph' in layer.name:
            hidd_layer = layer.name

    hidd_funcs = [K.function([model.get_layer('inp_nodes').input, K.learning_phase()],
                             [model.get_layer(dense_layer).output])]

    inps = [model.get_layer('inp_nodes').input]
    for i in range(n_layers - 1):
        inps.append(model.get_layer('inp_context_%d' % i).input)
        get_hidden = K.function(inps + [K.learning_phase()],
                                [model.get_layer(hidd_layer).get_output_at(i)])
        hidd_funcs.append(get_hidden)

    return hidd_funcs

def get_hiddens(model, x, contexts, hidd_funcs):
    hidds = []
    for i in range(len(hidd_funcs)):
        inps = [x] + contexts[:i]
        hidds.append(hidd_funcs[i](inps + [0])[0])

    return hidds