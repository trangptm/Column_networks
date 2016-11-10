import numpy
import sys
import prepare_data
args = prepare_data.arg_passing(sys.argv)
seed = args['-seed']
numpy.random.seed(seed)

from keras.optimizers import *
from create_model import *

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
    labels = labels[:, yidx : yidx+1]

def remove(y, not_ids):
    new_y = numpy.copy(y)
    for ids in not_ids:
        new_y[ids] = -1
    return new_y

if type == 'software':
    train_y = remove(labels, [test_ids])
    valid_y = remove(labels, [train_ids])
else:
    train_y = remove(labels, [valid_ids, test_ids])
    valid_y = remove(labels, [train_ids, test_ids])

n_classes = numpy.max(labels)
if n_classes > 1:
    n_classes += 1
    loss = multi_sparse_graph_loss
else:
    loss = graph_loss

if 'movie' in task:
    n_classes = -labels.shape[-1]

########################## BUILD MODEL ###############################################
print 'Building model ...'

# create model: n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared

if modelType == 'Highway':
    model = create_highway(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
elif modelType == 'Dense':
    model = create_dense(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
else:
    model = create_resNet(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)

model.summary()

# Full batch learning, learning rate should be large
lr = 0.01
opt = {'RMS': RMSprop(lr=lr, rho=0.9, epsilon=1e-8),
       'Adam': Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)}
model.compile(optimizer=opt[args['-opt']], loss=loss)

if 'movie' not in task:
    train_y = numpy.expand_dims(train_y, -1)
    valid_y = numpy.expand_dims(valid_y, -1)

json_string = model.to_json()
fModel = open('models/' + saving + '.json', 'w')
fModel.write(json_string)
fModel.close()

fParams = 'bestModels/' + saving + '.hdf5'
fResult = 'log/' + saving + '.txt'

f = open(fResult, 'w')
f.write('Training log:\n')
f.close()

saveResult = SaveResult([[feats, rel_list, rel_mask], labels, train_ids, valid_ids, test_ids],
                        task=task, fileResult=fResult, fileParams=fParams)

callbacks=[saveResult, NanStopping()]

his = model.fit([feats, rel_list, rel_mask], train_y,
                validation_data=([feats, rel_list, rel_mask], valid_y),
                nb_epoch=1000, batch_size=feats.shape[0], shuffle=False,
                callbacks=callbacks)