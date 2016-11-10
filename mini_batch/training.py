import numpy
import sys
import time
import prepare_data
args = prepare_data.arg_passing(sys.argv)
seed = args['-seed']
numpy.random.seed(seed)

from keras.optimizers import *
from keras.objectives import *
from create_model import *
from app_hidden import *

################################## LOAD DATA ##################################################
dataset = args['-data']
task = 'software'
if 'pubmed' in dataset:
    task = 'pubmed'
elif 'movie' in dataset:
    task = 'movie'

dataset = '../data/' + dataset + '.pkl.gz'
modelType = args['-model']
n_layers, dim = args['-nlayers'], args['-dim']
shared = args['-shared']
saving = args['-saving']
nmean = args['-nmean']
yidx = args['-y']

if 'dr' in args['-reg']: dropout = True
else: dropout = False
feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids = prepare_data.load_data(dataset)
print len(train_ids), len(valid_ids), len(test_ids)
labels = labels.astype('int64')
if task == 'movie':
    labels = labels[:, yidx]

n_classes = numpy.max(labels)
if n_classes > 1:
    n_classes += 1
    loss = sparse_categorical_crossentropy
else:
    loss = binary_crossentropy

########################## BUILD MODEL ###############################################
print 'Building model ...'

# create model: n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared

if modelType == 'Highway':
    model = create_highway(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
else:
    model = create_dense(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
model.summary()

lr = 0.001
opt = {'RMS': RMSprop(lr=lr, rho=0.9, epsilon=1e-8),
       'Adam': Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)}
#opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
#opt = Adagrad(lr=0.01, epsilon=1e-8)
#opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-8)
#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8
model.compile(optimizer=opt[args['-opt']], loss=loss)

json_string = model.to_json()
fModel = open('models/' + saving + '.json', 'w')
fModel.write(json_string)
fModel.close()

fParams = 'bestModels/' + saving + '.hdf5'
fResult = 'log/' + saving + '.txt'

f = open(fResult, 'w')
f.write('Training log:\n')
f.close()

past_hiddens = HiddenSaving(n_samples=feats.shape[0], n_layers=n_layers,
                            h_dim=dim, rel=rel_list, rel_mask=rel_mask)
nb_epoch = 1000
batch_size = int(args['-batch'])
mini_batch_gen = prepare_data.MiniBatchIds(len(train_ids), batch_size=batch_size)
n_batchs = len(train_ids) // batch_size
if len(train_ids) % batch_size > 0: n_batchs += 1

valid_x, valid_y = feats[valid_ids], labels[valid_ids]
test_x, test_y = feats[test_ids], labels[test_ids]

saveResult = SaveResult(task=task, fileResult=fResult, fileParams=fParams)
callbacks = [saveResult, NanStopping()]
hidd_funcs = get_hidden_funcs(model, n_layers)

for epoch in xrange(nb_epoch):
    start = time.time()
    for i in xrange(n_batchs):
        mini_batch_ids = train_ids[mini_batch_gen.get_mini_batch_ids(i)]
        train_x = feats[mini_batch_ids]
        train_y = labels[mini_batch_ids]
        context = past_hiddens.get_context(mini_batch_ids)

        # in the last mini-batch of an epoch
        # compute the context for valid and test, save result
        # update hidden states of valid and test in past_hiddens.
        if i == n_batchs - 1:
            valid_context = past_hiddens.get_context(valid_ids)
            test_context = past_hiddens.get_context(test_ids)
            valid_data = [[valid_x] + valid_context, valid_y]
            saveResult.update_data([valid_x] + valid_context, valid_y, [test_x] + test_context, test_y)

            for x, ct, ids in zip([valid_x, test_x], [valid_context, test_context], [valid_ids, test_ids]):
                new_hiddens = get_hiddens(model, x, ct, hidd_funcs)
                past_hiddens.update_hidden(ids, new_hiddens)
        else:
            valid_data = None

        his = model.fit([train_x] + context, numpy.expand_dims(train_y, -1),
                        validation_data=valid_data, verbose=0,
                        nb_epoch=1, batch_size=train_x.shape[0], shuffle=False,
                        callbacks=callbacks)

        # update hidden states of train in a mini-batch after gradient updating
        new_train_hiddens = get_hiddens(model, train_x, context, hidd_funcs)
        past_hiddens.update_hidden(mini_batch_ids, new_train_hiddens)

    end = time.time()
    print 'epoch %d, runtime: %.1f' % (epoch, end - start)
    if model.stop_training:
        break