Training the column network in mini batch scheme

Instead of directly computing the neighbor hidden state, the model uses their values after their latest gradient update. This is an approximation method for the full-batch learning. The approach is much faster and appropriate for large-scale networks
