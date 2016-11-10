import numpy
from theano import tensor
from keras.callbacks import *
from keras import objectives
from keras.layers import *
from keras.models import Model
from sklearn import metrics
from keras.constraints import *
from keras.layers.advanced_activations import *
from graph_layers import *

class SaveResult(Callback):
    '''
    Compute result after each epoch. Return a log of result
    Arguments:
        data_x, data_y, metrics
    '''

    def __init__(self, data=None, task='software', fileResult='', fileParams='', minPatience=5, maxPatience=20):
        super(SaveResult, self).__init__()

        self.x = None
        self.y = None
        self.valid_ids = None
        self.test_ids = None
        self.do_test = False

        f = open(fileResult, 'a')
        aucstr = 'auc'
        if len(data) >= 4:
            self.x, self.y = data[0], data[1]
            self.train_ids = data[2]
            self.valid_ids = data[3]
            if numpy.max(self.y) > 1: aucstr = 'maf1'
            f.write('epoch\tloss\tv_loss\t|\ttr_' + aucstr + '\ttr_f1\ttr_pre\ttr_rec\t|\tv_auc\tv_f1\tv_pre\tv_rec\t|')
        if len(data) == 5:
            self.test_ids = data[4]
            f.write('\tt_' + aucstr + '\tt_f1\tt_pre\tt_rec')
            self.do_test = True
        f.write('\n')
        f.close()

        self.bestResult = 0.0
        self.bestEpoch = 0
        # wait to divide the learning rate. if reach the maxPatience -> stop learning
        self.wait = 0
        self.patience = minPatience
        self.maxPatience = maxPatience

        self.task = task
        self.fileResult = fileResult
        self.fileParams = fileParams

    def _compute_result(self, y_pred, y_true, ids):
        y_true = y_true[ids]
        y_pred = y_pred[ids]

        if 'software' in self.task:
            y_pred = 1.0 - y_pred[:, 0]
            nonzero_ids = numpy.nonzero(y_true)[0]
            y_true[nonzero_ids] = 1
        if 'movie' in self.task:
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

        if numpy.max(y_true) == 1:
            fp, tp, thresholds = metrics.roc_curve(y_true, y_pred)
            auc = metrics.auc(fp, tp)
            y_pred = numpy.round(y_pred)
            average = 'binary'
        else:
            y_pred = numpy.argmax(y_pred, axis=1)
            average = 'micro'
            auc = metrics.f1_score(y_true, y_pred, average='macro')

        if numpy.isnan(y_pred).any(): return 0.0, 0.0, 0.0, 0.0

        # metric can be 'f1_binary', 'f1_micro', 'f1_macro' (for multi-classes)
        call = {'f1': metrics.f1_score,
                'recall': metrics.recall_score,
                'precision': metrics.precision_score}

        pre = call['precision'](y_true, y_pred, average=average)
        rec = call['recall'](y_true, y_pred, average=average)
        f1 = call['f1'](y_true, y_pred, average=average)
        return auc, f1, pre, rec


    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x, batch_size=self.x[0].shape[0])
        tr_auc, tr_f1, tr_pre, tr_rec = self._compute_result(y_pred, self.y, self.train_ids)
        v_auc, v_f1, v_pre, v_rec = self._compute_result(y_pred, self.y, self.valid_ids)

        f = open(self.fileResult, 'a')
        f.write('%d\t%.4f\t%.4f\t|\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (epoch, logs['loss'], logs['val_loss'], tr_auc, tr_f1, tr_pre, tr_rec))
        f.write('\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (v_auc, v_f1, v_pre, v_rec))
        if self.do_test:
            t_auc, t_f1, t_pre, t_rec = self._compute_result(y_pred, self.y, self.test_ids)
            f.write('\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (t_auc, t_f1, t_pre, t_rec))

        if v_f1 > self.bestResult:
            self.bestResult = v_f1
            self.bestEpoch = epoch
            self.model.save_weights(self.fileParams, overwrite=True)
            self.wait = 0
        f.write('  Best result at epoch %d\n' % self.bestEpoch)
        f.close()

        if v_f1 < self.bestResult:
            self.wait += 1
            if self.wait == self.patience:
                self.wait = 0
                self.patience += 5

                lr = K.get_value(self.model.optimizer.lr) / 2.0
                K.set_value(self.model.optimizer.lr, lr)
                print ('New learning rate: %.4f', K.get_value(self.model.optimizer.lr))
                if self.patience > self.maxPatience:
                    self.model.stop_training = True

class LRScheduler(Callback):
    def __init__(self, n_epoch):
        super(LRScheduler, self).__init__()
        self.max_epoch = n_epoch
        self.n_epoch = n_epoch

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'

        if self.n_epoch == 0:
            self.n_epoch = self.max_epoch
            lr = self.model.optimizer.lr / 2.0
            self.model.optimizer.lr = lr
        else:
            self.n_epoch -= 1
            self.max_epoch = min(10, self.max_epoch - 1)

class NanStopping(Callback):
    def __init__(self):
        super(NanStopping, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        for k in logs.values():
            if numpy.isnan(k):
                self.model.stop_training = True

def graph_loss(y_true, y_pred):
    ids = tensor.nonzero(y_true + 1)[0]
    y_true = y_true[ids]
    y_pred = y_pred[ids]
    return tensor.mean(tensor.nnet.binary_crossentropy(y_pred, y_true))

def multi_sparse_graph_loss(y_true, y_pred):
    ids = tensor.nonzero(y_true[:,0] + 1)[0]
    y_true = y_true[ids]
    y_pred = y_pred[ids]

    return tensor.mean(objectives.sparse_categorical_crossentropy(y_true, y_pred))

def create_highway(n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared, nmean=1, dropout=True, rel_carry=True):
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    trans_bias = - n_layers * 0.1

    shared_highway = GraphHighway(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                                  init=init, activation=act, transform_bias=trans_bias)

    def highway(shared):
        if shared == 1: return shared_highway
        return GraphHighway(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                            init=init, activation=act, transform_bias=trans_bias)

    #x, rel, rel_mask
    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    inp_rel = Input(shape=(n_rel, n_neigh), dtype='int64', name='inp_rel')
    inp_rel_mask = Input(shape=(2, n_rel, n_neigh), dtype='float32', name='inp_rel_mask')

    hidd_nodes = Dense(output_dim=hidden_dim, input_dim=input_dim, activation=act)(inp_nodes)
    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = highway(shared)([hidd_nodes, inp_rel, inp_rel_mask])
        if shared == 0 and i % 5 == 2:
            hidd_nodes = Dropout(0.5)(hidd_nodes)

    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)
    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)

    model = Model(input=[inp_nodes, inp_rel, inp_rel_mask], output=[top_nodes])

    return model

def create_resNet(n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared, nmean, dropout=True):
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    shared_graph = GraphDense(input_dim=hidden_dim, output_dim=hidden_dim, init=init,
                              n_rel=n_rel, mean=nmean, activation=act)
    shared_dense = Dense(output_dim=hidden_dim, input_dim=hidden_dim)

    def residual(shared):
        if shared == 1: return shared_graph, shared_dense
        graph = GraphDense(input_dim=hidden_dim, output_dim=hidden_dim, init=init,
                              n_rel=n_rel, mean=nmean, activation=act)
        dense = Dense(output_dim=hidden_dim, input_dim=hidden_dim)
        return graph, dense

    def ResBlock(inputs, graph_layer, dense_layer):
        inp = inputs[0]
        #inputs[0] = BatchNormalization()(inputs[0])
        hidd = graph_layer(inputs)
        hidd = dense_layer(hidd)

        block = merge([inp, hidd], mode='sum')
        return block

    #x, rel, rel_mask
    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    inp_rel = Input(shape=(n_rel, n_neigh), dtype='int64', name='inp_rel')
    inp_rel_mask = Input(shape=(2, n_rel, n_neigh), dtype='float32', name='inp_rel_mask')

    hidd_nodes = Dense(output_dim=hidden_dim, input_dim=input_dim, activation=act)(inp_nodes)
    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    for i in range(n_layers):
        graph_layer, dense_layer = residual(shared)
        hidd_nodes = ResBlock([hidd_nodes, inp_rel, inp_rel_mask], graph_layer, dense_layer)

    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)
    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)
    model = Model(input=[inp_nodes, inp_rel, inp_rel_mask], output=[top_nodes])
    return model

def create_dense(n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared, nmean=1, dropout=True):
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    inp_rel = Input(shape=(n_rel, n_neigh), dtype='int64', name='inp_rel')
    inp_rel_mask = Input(shape=(2, n_rel, n_neigh), dtype='float32', name='inp_rel_mask')

    hidd_nodes = Dense(input_dim=input_dim, output_dim=hidden_dim, init=init, activation=act)(inp_nodes)
    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = GraphDense(input_dim=hidden_dim, output_dim=hidden_dim, init=init,
                                n_rel=n_rel, mean=nmean, activation=act)([hidd_nodes, inp_rel, inp_rel_mask])
        if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)
    model = Model(input=[inp_nodes, inp_rel, inp_rel_mask], output=[top_nodes])

    return model

def create_highway_noRel(n_layers, hidden_dim, input_dim, n_classes, shared=1, dropout=True):
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    trans_bias = - n_layers * 0.1

    shared_highway = Highway(input_dim=hidden_dim, init=init, activation=act, transform_bias=trans_bias)

    def highway(shared):
        if shared == 1: return shared_highway
        return Highway(input_dim=hidden_dim, init=init, activation=act, transform_bias=trans_bias)

    #x, rel, rel_mask
    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    hidd_nodes = Dense(output_dim=hidden_dim, input_dim=input_dim, activation=act)(inp_nodes)
    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = highway(shared)(hidd_nodes)

    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)
    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)

    model = Model(input=inp_nodes, output=[top_nodes])

    return model