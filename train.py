import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import CDLLD
from data import Multi_view_data
import warnings
import ldl_metrics as ldlm

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lambda-epochs', type=int, default=1, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()
    args.data_name = 'SJA_c_split_binary'
    args.data_path = 'datasets/' + args.data_name

    args.views = 1

    train_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=True), batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=False), batch_size=args.batch_size, shuffle=False)
    N_mini_batches = len(train_loader)


    args.dims = [[Multi_view_data(args.data_path, train=False).lenx()]]
    args.classes = classes = Multi_view_data(args.data_path, train=False).leny() - 1


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, loss = model(data, torch.nn.functional.softmax(target[:, :classes]), epoch)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        Yhat = torch.empty(0, classes).cuda()
        GT = torch.empty(0, classes).cuda()
        for batch_idx, (data, target) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            with torch.no_grad():
                target = Variable(target.cuda())
                evidences, loss = model(data, torch.nn.functional.softmax(target[:, :classes]), epoch)
                #_, predicted = torch.max(evidences.data, 1)
                #print(evidences[0].size())
                if batch_idx == 0:
                    Yhat = evidences[0]
                    GT = target
                else:
                    Yhat = torch.cat((Yhat, evidences[0]), 0)
                    GT = torch.cat((GT, target), 0)
                #print(torch.nn.functional.normalize(target[:, :classes], p=1, dim=1))
                loss_meter.update(loss.item())

        S = torch.sum((Yhat + 1), dim=1, keepdim=True)

        b = Yhat / (S.expand(Yhat.shape))
        u = args.classes / S


        Yhat = torch.cat((b, u), 1)


        GT = GT.cpu().numpy()
        Yhat = Yhat.cpu().numpy()




        metrics = np.array([ldlm.Cheby(GT, Yhat), ldlm.Clark(GT, Yhat),
                            ldlm.KL_div(GT, Yhat), ldlm.Cosine(GT, Yhat)])

        return loss_meter.avg, correct_num / data_num, metrics

    K = 2

    wdd = {'Natural_Scene_split_binary': 1e-5, 'SJA_c_split_binary': 1, 'Yeast_alpha_split_binary': 1e-5}

    for k in range(1, K + 1):

        model = CDLLD(args.classes, args.views, args.dims, args.lambda_epochs)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=wdd[args.data_name])

        model.cuda()


        for epoch in range(1, args.epochs + 1):
            train(epoch)


        test_loss, acc, metrics = test(args.epochs)
        if k == 1:
            overall_metrics = metrics
        else:
            overall_metrics = np.column_stack((overall_metrics, metrics))

    mean_m = np.mean(overall_metrics, axis=1)
    std_m = np.std(overall_metrics, axis=1)

    print('==================================')
    print('Mean in [Cheby Clark KL Cosine]:')
    print(mean_m)
    print('==================================')
    print('Std in [Cheby Clark KL Cosine]:')
    print(std_m)
    print('==================================')


