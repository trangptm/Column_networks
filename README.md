# Column_networks
A novel deep learning model for collective classification in multi-relational domains

Training: enter either full_batch or mini_batch folder

python training.py [-opt option_val]

-opt      default_option (using / for separating possible options, ... is any value)

-model    Highway (Highway/Dense)

-data     pubmed (pubmed/BoW_movielens)

-dim      50 (the hidden dimension of the column network, this hyper-param should be tuned)

-nlayers  10 (the number of highway layer, should be 10)

-nmean    1 (the constant z in the paper, should be between 1 and the number of relation types)

-reg      dr (dr/x. dr means dropout and x means no regularization)

-shared   1 (0/1. 0: no parameter sharing among highway layers)

-opt      RMS (Adam/RMS. The optimizer)

-seed     1234 (any value)

-batch    100 (batch size in the mini_batch training only)

-y        1 (in the case of multi-label, e.g., movielens, each label is trained separetely. This option is for choosing what label is trained)
