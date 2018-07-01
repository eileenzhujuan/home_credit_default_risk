# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, merge, Reshape, Dropout, Input, Flatten, Concatenate, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping


def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Embedding Neural Network ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def display_precision_recall(y_, oof_preds_, folds_idx_):
    # Plot ROC curves 
    plt.figure(figsize=(6, 6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Embedding Neural Network Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.show()


def build_embedding_network(len_embed_cols, x_dim):
    model_out = []
    model_in = []

    # usual rule of embedding size = min(50, number of categories/2)
    for dim in len_embed_cols:
        input_dim = Input(shape=(1,), dtype='int32')
        embed_dim = Embedding(dim, dim // 2, input_length=1)(input_dim)
        embed_dim = Dropout(0.25)(embed_dim)
        embed_dim = Reshape((dim // 2,))(embed_dim)
        model_out.append(embed_dim)
        model_in.append(input_dim)

    input_num = Input(shape=(x_dim,), dtype='float32')
    outputs = Concatenate(axis=1)([*model_out, input_num])

    outputs = (Dense(128))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (BatchNormalization())(outputs)
    outputs = (Dropout(.35))(outputs)
    outputs = (Dense(64))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    outputs = (Dense(32))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    outputs = (Dense(1))(outputs)
    outputs = (Activation('sigmoid'))(outputs)

    model = Model([*model_in, input_num], outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test


if __name__ == '__main__':
    train = pd.read_csv('../data/preprocess_train.csv')
    test = pd.read_csv('../data/preprocess_test.csv')
    X_train, y_train = train.iloc[:, 2:], train.TARGET
    X_test = test.iloc[:, 1:]
    col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if X_train[c].dtype == object}
    # nb_numeric   = len(X_train.columns) - len(col_vals_dict)
    # nb_categoric = len(col_vals_dict)
    # print('Number of Numerical features:', nb_numeric)
    # print('Number of Categorical features:', nb_categoric)

    # Generator to parse the cat
    generator = (c for c in X_train.columns if X_train[c].dtype == object)

    # Label Encoder
    for c in generator:
        lbl = LabelEncoder()
        lbl.fit(list(X_train[c].values) + list(X_test[c].values))
        X_train[c] = lbl.transform(list(X_train[c].values))
        X_test[c] = lbl.transform(list(X_test[c].values))

    embed_cols = []
    len_embed_cols = []
    for c in col_vals_dict:
        if len(col_vals_dict[c]) > 2:
            embed_cols.append(c)
            len_embed_cols.append(len(col_vals_dict[c]))
            print(c + ': %d values' % len(col_vals_dict[c]))  # look at value counts to know the embedding dimensions

    print('\n Number of embed features :', len(embed_cols))

    # Select the numeric features
    num_cols = [x for x in X_train.columns if x not in embed_cols]

    # Impute missing values in order to scale
    X_train[num_cols] = X_train[num_cols].fillna(value=0)
    X_test[num_cols] = X_test[num_cols].fillna(value=0)

    # Fit the scaler only on train data
    scaler = MinMaxScaler().fit(X_train[num_cols])
    X_train.loc[:, num_cols] = scaler.transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    K = 5
    runs_per_fold = 1
    n_epochs = 250
    patience = 10

    cv_aucs = []
    full_val_preds = np.zeros(np.shape(X_train)[0])
    y_preds = np.zeros((np.shape(X_test)[0], K))

    kfold = StratifiedKFold(n_splits=K,
                            shuffle=True, random_state=1)

    for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

        X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
        y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]

        X_test_f = X_test.copy()

        # Shuffle data
        idx = np.arange(len(X_train_f))
        np.random.shuffle(idx)
        X_train_f = X_train_f.iloc[idx]
        y_train_f = y_train_f.iloc[idx]

        # preprocessing
        proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)

        # track oof prediction for cv scores
        val_preds = 0

        for j in range(runs_per_fold):
            NN = build_embedding_network(len_embed_cols, x_dim=176)

            # Set callback functions to early stop training and save the best model so far
            callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]

            NN.fit(proc_X_train_f, y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=2, callbacks=callbacks,
                   validation_data=(proc_X_val_f, y_val_f))

            val_preds += NN.predict(proc_X_val_f)[:, 0] / runs_per_fold
            y_preds[:, i] += NN.predict(proc_X_test_f)[:, 0] / runs_per_fold

        full_val_preds[outf_ind] += val_preds

        cv_auc = roc_auc_score(y_val_f.values, val_preds)
        cv_aucs.append(cv_auc)
        print('\nFold %i prediction cv AUC: %.5f\n' % (i, cv_auc))

    print('Mean out of fold AUC: %.5f' % np.mean(cv_auc))
    print('Full validation AUC: %.5f' % roc_auc_score(y_train.values, full_val_preds))
