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

    def __init__(self, task='software', fileResult='', fileParams='', minPatience=5, maxPatience=20):
        super(SaveResult, self).__init__()

        self.valid_x = None
        self.valid_y = None
        self.test_x = None
        self.test_y = None
        self.do_test = False
        self.save_result = False

        self.bestResult = 0.0
        self.bestEpoch = 0
        self.n_epoch = -1
        # wait to divide the learning rate. if reach the maxPatience -> stop learning
        self.wait = 0
        self.patience = minPatience
        self.maxPatience = maxPatience

        self.task = task
        self.fileResult = fileResult
        self.fileParams = fileParams

    def update_data(self, valid_x, valid_y, test_x=None, test_y=None):
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.save_result = True
        if self.test_x is not None:
            self.do_test = True

    def _compute_result(self, x, y_true):
        y_pred = self.model.predict(x, batch_size=x[0].shape[0])
        if 'software' in self.task:
            y_pred = 1.0 - y_pred[:, 0]
            nonzero_ids = numpy.nonzero(y_true)[0]
            y_true[nonzero_ids] = 1

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
        if not self.save_result: return
        self.n_epoch += 1
        self.save_result = False

        v_auc, v_f1, v_pre, v_rec = self._compute_result(self.valid_x, self.valid_y)

        f = open(self.fileResult, 'a')
        f.write('%d\t%.4f\t%.4f\t|\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (self.n_epoch, logs['loss'], logs['val_loss'], v_auc, v_f1, v_pre, v_rec))
        if self.do_test:
            t_auc, t_f1, t_pre, t_rec = self._compute_result(self.test_x, self.test_y)
            f.write('\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (t_auc, t_f1, t_pre, t_rec))

        if v_f1 > self.bestResult:
            self.bestResult = v_f1
            self.bestEpoch = self.n_epoch
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
    contexts = [Input(shape=(n_rel, hidden_dim), dtype='float32', name='inp_context_%d' % i)
                for i in range(n_layers)]

    hidd_nodes = Dense(output_dim=hidden_dim, input_dim=input_dim, activation=act)(inp_nodes)
    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = highway(shared)([hidd_nodes, contexts[i]])
        if shared == 0 and i % 5 == 2:
            hidd_nodes = Dropout(0.5)(hidd_nodes)

    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)
    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)

    model = Model(input=[inp_nodes] + contexts, output=[top_nodes])

    return model

def create_dense(n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared, nmean=1, dropout=True):
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    contexts = [Input(shape=(n_rel, hidden_dim), dtype='float32', name='inp_context_%d' % i)
                for i in range(n_layers)]

    hidd_nodes = Dense(input_dim=input_dim, output_dim=hidden_dim, init=init, activation=act)(inp_nodes)
    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = GraphDense(input_dim=hidden_dim, output_dim=hidden_dim, init=init,
                                n_rel=n_rel, mean=nmean, activation=act)([hidd_nodes, contexts[i]])
        if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)
    model = Model(input=[inp_nodes, contexts], output=[top_nodes])

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