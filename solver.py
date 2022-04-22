import matplotlib.pyplot as plt

from model import PrefNet
from dataset import pref_dataset, utility_dataset
from utils import *
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import al_acquisition
import bo_acquisition
import torchbnn as bnn
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solver_nn(train0, query0, test, query_bo, n_acq_al, n_acq_bo, al_acq, bo_acq):
    """
    controller for BNN and PBNN
    """
    train = train0.copy()
    query = query0.copy()

    if al_acq is None:
        model = PrefNet(train['x_duels'][0][0].size).to(device).double()
        min_list = bo_nn(model, query_bo, n_acq_bo, bo_acq)
    else:
        model, train_al = apl_nn(train, query, test, n_acq_al, al_acq)
        # initialize the BO output layers using expert output layer
        model.fc5.load_state_dict(model.fc4.state_dict())
        min_list = bo_nn(model, query_bo, n_acq_bo, bo_acq, train_al=train_al)

    return min_list


def apl_nn(train, query, test, n_acq_al, al_acq):
    print("Start active learning with preference data")
    model = update_nn_pref(train['x_duels'], train['pref'], model=None)
    model.train()
    al_function = al_acquisition.choose_criterion(al_acq)
    expert_acc = 0
    for i in range(n_acq_al):
        print("Acquisition time: " + str(i+1))
        query_index = al_function(model, train, query)
        print("point 0: [{:.2f}, {:.2f}] point 1: [{:.2f}, {:.2f}]".format(query['x_duels'][query_index, 0][0],
                                                                           query['x_duels'][query_index, 0][1],
                                                                           query['x_duels'][query_index, 1][0],
                                                                           query['x_duels'][query_index, 1][1]))

        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        x = [query['x_duels'][query_index, 0][0], query['x_duels'][query_index, 1][0]]
        y = [query['x_duels'][query_index, 0][1], query['x_duels'][query_index, 1][1]]
        titles = ["point 0", "point 1"]

        ax.scatter(x, y, [0, 0], c='b')
        for j, txt in enumerate(titles):
            ax.text(x[j], y[j], 0, txt)
        ax.set_title("3D plot")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f')
        ax.set_zlim3d(0, 6)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-2, 2)
        plt.show()
        # print("ground truth: " + str(query['pref'][query_index]))
        pref = input("Which point do you think is larger? (0 or 1):")

        if int(pref) != 1 and int(pref) != 0:
            pref = input("Wrong input! Try again (0 or 1):")
        if int(pref) == int(query['pref'][query_index]):
            expert_acc += 1
        print("How biased our expert is: {:.2%}".format(expert_acc / (i+1)))
        train['x_duels'] = np.vstack((train['x_duels'], query['x_duels'][[query_index], :]))
        train['pref'] = np.hstack((train['pref'], int(pref)))
        # train['pref'] = np.hstack((train['pref'], query['pref'][query_index]))

        query['x_duels'] = np.delete(query['x_duels'], query_index, axis=0)
        query['pref'] = np.delete(query['pref'], query_index)

        model = update_nn_pref(train['x_duels'], train['pref'], model=model)

        # print("{} query of preferential active learning".format(i+1))
        compute_nn_acc(model, test)
        plt.close()
        # plt.figure(figsize=(7, 5.3))
        # csfont = {'fontname': 'Times New Roman'}
        # x, y, _, query_x, query_y = get_forrester_data(1)
        # x = torch.tensor(x)
        # out_for_plot = model.forward_once(x)
        # plt.plot(x.detach().numpy(), y, color="red", linewidth=3, label="$f(\mathbf{x})$")
        # plt.plot(x.detach().numpy(), out_for_plot.detach().numpy(), color="black", linestyle='-.', linewidth=3, label="$\~{g}(\mathbf{x})$")
        # plt.legend(loc=2, prop={'size': 20})
        # plt.xlabel("$\mathbf{x}$", size=18)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.title("Styblinski-Tang function", **csfont, fontsize=24)
        # # np.save("npy/"+str(i)+"forr", out_for_plot.detach().numpy())
        # plt.show()

    return model, train


def compute_nn_acc(model, test):
    """
    compute the preference accuracy for neural network
    :param model: current model
    :param test: test set
    :return: prediction accuracy
    """
    model.eval()
    x_test = test['x_duels']
    pref_test = test['pref']
    acc = 0
    n_mc = 2
    for i in range(len(x_test)):
        x1 = torch.tensor(x_test[i][0])
        x2 = torch.tensor(x_test[i][1])
        pref = pref_test[i]
        out = torch.zeros((n_mc, 2))
        for n in range(n_mc):
            out[n, 0], out[n, 1] = model(x1, x2)
        pred = torch.mean(out, dim=0)

        out1 = pred[0]
        out2 = pred[1]
        if pref == 1 and out1 < out2:
            acc += 1
        if pref == 0 and out1 > out2:
            acc += 1
    acc = acc / len(x_test)
    print("Expert surrogate model accuracy (compared with f)", acc)
    print("")
    return acc


def update_nn_pref(x_duels, pref, model=None):
    pref_set = pref_dataset(x_duels, pref)
    pref_train_loader = DataLoader(pref_set, batch_size=10, shuffle=True, drop_last=False)
    pref_net = PrefNet(x_duels[0][0].size).to(device) if model is None else model
    pref_net.double()

    criterion = torch.nn.NLLLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = torch.optim.Adam(pref_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0001, T_max=20)

    for epoch in range(100):
        pref_net.train()
        train_loss = 0
        # train with preference pairs
        for idx, data in enumerate(pref_train_loader):
            x1 = data['x1']
            x2 = data['x2']

            pref = data['pref'].long()
            x1, x2, pref = x1.to(device), x2.to(device), pref.to(device)
            optimizer.zero_grad()
            output1, output2 = pref_net(x1, x2)

            output = F.log_softmax(torch.hstack((output1, output2)), dim=1)

            loss = criterion(output, pref) + 0.1 * kl_loss(pref_net)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        # print('[Epoch : %d] loss: %.3f' % (epoch + 1, train_loss / len(pref_train_loader)))
    return pref_net


def bo_nn(model, query, n_acq_bo, bo_acq, **kwargs):
    test = query.copy()
    min_list = np.zeros(n_acq_bo, )
    bo_function = bo_acquisition.choose_criterion(bo_acq)
    y_best = 1000

    print("Start Bayesian optimization with utility function")

    if "train_al" in kwargs.keys():
        start_index = bo_function(query, model, y_best)
        train_x = query['x'][[start_index], :]
        train_y = query['y'][[start_index]]
        query['x'] = np.delete(query['x'], start_index, axis=0)
        query['y'] = np.delete(query['y'], start_index)
        model = update_nn_multi(train_x, train_y, kwargs['train_al']['x_duels'], kwargs['train_al']['pref'], model, 1)
    else:
        start_index = np.random.choice(len(query['x']), 1, replace=False)
        train_x = query['x'][start_index, :]
        train_y = query['y'][start_index]
        query['x'] = np.delete(query['x'], start_index, axis=0)
        query['y'] = np.delete(query['y'], start_index)
        model = update_nn_reg(train_x, train_y, model)

    # delete later
    # show_shape(model)

    for i in range(n_acq_bo):
        query_index = bo_function(query, model, y_best)
        train_x = np.vstack((train_x, query['x'][[query_index], :]))
        train_y = np.hstack((train_y, query['y'][query_index]))

        query['x'] = np.delete(query['x'], query_index, axis=0)
        query['y'] = np.delete(query['y'], query_index)

        if "train_al" in kwargs.keys():
            model = update_nn_multi(train_x, train_y, kwargs['train_al']['x_duels'], kwargs['train_al']['pref'], model, i)
        else:
            model = update_nn_reg(train_x, train_y, model)

        # pred_best = find_min_nn(model, test)
        pred_best = min(train_y)
        y_best = pred_best if pred_best < y_best else y_best
        min_list[i] = y_best

        print("{} query of Bayesian optimization, min value {}".format(i + 1, min_list[i]))

        # delete later
        # show_shape(model)

    return min_list


def update_nn_reg(x, y, model):
    inducing_set = utility_dataset(x, y)
    inducing_train_loader = DataLoader(inducing_set, batch_size=10, shuffle=True, drop_last=False)

    inducing_optim = torch.optim.Adam(model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(inducing_optim, eta_min=0.001, T_max=20)

    inducing_criterion = torch.nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    model.train()
    for epoch in range(200):
        loss = torch.zeros(1)
        for idx, data in enumerate(inducing_train_loader):
            inducing_x = data['x'].to(device)
            inducing_y = data['y'].to(device)
            inducing_optim.zero_grad()
            pred = model.forward_bo(inducing_x)
            pred = pred.flatten()
            # loss[0] += inducing_criterion(pred, inducing_y)
            loss = inducing_criterion(pred, inducing_y) + 0.1 * kl_loss(model)
        # loss += 0.1 * kl_loss(model).reshape(-1)
            loss.backward()
            inducing_optim.step()
            # scheduler.step()
    return model


def update_nn_multi(x, y, x_duels, y_pref, model, weight):
    inducing_set = utility_dataset(x, y)
    inducing_train_loader = DataLoader(inducing_set, batch_size=5, shuffle=True, drop_last=False)

    pref_set = pref_dataset(x_duels, y_pref)
    pref_train_loader = DataLoader(pref_set, batch_size=10, shuffle=True, drop_last=False)

    criterion = torch.nn.NLLLoss()
    inducing_criterion = torch.nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(200):

        pref_loss = 0
        reg_loss = 0
        # train with preference pairs
        for idx, data in enumerate(pref_train_loader):
            x1 = data['x1']
            x2 = data['x2']
            pref = torch.tensor([0 if data['pref'][i] < 0.5 else 1 for i in range(len(data['pref']))])

            x1, x2, pref = x1.to(device), x2.to(device), pref.to(device)
            optimizer.zero_grad()
            output1, output2 = model(x1, x2)
            output = F.log_softmax(torch.hstack((output1, output2)), dim=1)

            pref_loss += criterion(output, pref) + 0.1 * kl_loss(model)

        # train with regression data
        for idx, data in enumerate(inducing_train_loader):
            inducing_x = data['x'].to(device)
            inducing_y = data['y'].to(device)
            # optimizer.zero_grad()
            pred = model.forward_bo(inducing_x)
            pred = pred.flatten()

            reg_loss += inducing_criterion(pred, inducing_y) + 0.1 * kl_loss(model)
        loss = (0.95 ** weight) * pref_loss + reg_loss
        loss.backward()
        optimizer.step()
    return model


def find_min_nn(model, test):
    model.eval()
    x = test['x']
    y = test['y']

    x = torch.tensor(x)
    pred = model.forward_bo(x)

    min_value = y[torch.argmin(pred)]
    return min_value


def show_shape(model):
    x, y, _, query_x, query_y = get_forrester_data(1)
    x = torch.tensor(x)
    out_for_plot = model.forward_bo(x)
    plt.plot(x.detach().numpy(), out_for_plot.detach().numpy())
    plt.plot(x.detach().numpy(), y, color="red")
    plt.show()






