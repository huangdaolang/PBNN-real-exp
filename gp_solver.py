from GPro.preference import ProbitPreferenceGP
import copy
import al_acquisition
import numpy as np


def train_gp(x_duels, pref, model=None):
    x_train = []
    M_train = []
    for i in range(len(x_duels)):
        x_train.append(x_duels[i][0])
        x_train.append(x_duels[i][1])
        M_train.append([2 * i, 2 * i + 1]) if pref[i] == 1 else M_train.append([2 * i + 1, 2 * i])

    gpr = ProbitPreferenceGP()
    gpr.fit(x_train, M_train, f_prior=None)
    return gpr


def compute_gp_acc(model, test):
    x_test = test['x_duels']
    pref_test = test['pref']
    acc = 0
    for i in range(len(x_test)):
        x1 = x_test[i][0]
        x2 = x_test[i][1]
        pref = pref_test[i]
        out1 = model.predict(x1.reshape(1, -1))
        out2 = model.predict(x2.reshape(1, -1))
        if pref == 1 and out1 > out2:
            acc += 1
        if pref == 0 and out1 < out2:
            acc += 1
    acc = acc / len(x_test)
    print("gp", acc)
    return acc


def active_train_gp(model, train0, query0, test, n_acq, al_criterion):
    train = train0.copy()
    query = query0.copy()
    model = copy.deepcopy(model)
    acc = np.zeros(n_acq + 1, )
    acc[0] = compute_gp_acc(model, test)

    al_function = al_acquisition.choose_criterion(al_criterion)

    for i in range(n_acq):
        query_index = al_function(model, train, query, test)
        pref_q = query['pref'][query_index]

        train['x_duels'] = np.vstack((train['x_duels'], query['x_duels'][[query_index], :]))
        train['pref'] = np.hstack((train['pref'], pref_q))

        query['x_duels'] = np.delete(query['x_duels'], query_index, axis=0)
        query['pref'] = np.delete(query['pref'], query_index)

        model = train_gp(train['x_duels'], train['pref'], model)

        acc[i+1] = compute_gp_acc(model, test)

    return acc
