import numpy as np
import torch
from scipy import interpolate
from tqdm import tqdm

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)

    acc = float(tp +tn ) /dist.shape[0]
    return tpr, fpr, acc

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def ACER(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    apcer = fp / (tn*1.0 + fp*1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer,tp, fp, tn,fn

def TPR_FPR( dist, actual_issame, fpr_target = 0.001):
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []

    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)

    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)

    thresholds = np.arange(0.0, 1.0, 0.001)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)
            TPR = tp / (tp*1.0 + fn*1.0)

        fpr[threshold_idx] = FPR

    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds, kind= 'slinear')
        threshold = f(fpr_target)
    else:
        threshold = 0.0

    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    FPR = fp / (fp * 1.0 + tn * 1.0)
    TPR = tp / (tp * 1.0 + fn * 1.0)

    print(str(FPR)+' '+str(TPR))
    return FPR,TPR

import torch.nn.functional as F
def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob

def do_valid( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []

    for input, truth in test_loader:
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]

def do_valid_test( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    correct_repalys = []
    correct_prints = []
    correct_faces = []
    probs = []
    prob_repalys = []
    prob_prints = []
    prob_faces = []
    labels = []
    repalylabels = []
    printlabels = []
    facelabels = []

    for i, (input, truth, repalyerr, printerr, faceerr) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cpu()
        truth = truth.cpu()
        repalyerr = repalyerr.cpu()
        printerr = printerr.cpu()
        faceerr = faceerr.cpu()

        with torch.no_grad():
            logit = net(input)
            logit_0 = logit[0].cpu()
            logit_1 = logit[1].cpu()
            logit_2 = logit[2].cpu()
            logit_3 = logit[3].cpu()
            logit_0 = logit_0.view(b,n,2)
            logit_1 = logit_1.view(b,n,2)
            logit_2 = logit_2.view(b,n,2)
            logit_3 = logit_3.view(b,n,2)
            logit_0 = torch.mean(logit_0, dim = 1, keepdim = False)
            logit_1 = torch.mean(logit_1, dim = 1, keepdim = False)
            logit_2 = torch.mean(logit_2, dim = 1, keepdim = False)
            logit_3 = torch.mean(logit_3, dim = 1, keepdim = False)

            truth = truth.view(logit_0.shape[0])
            repalyerr = repalyerr.view(logit_1.shape[0])
            printerr = printerr.view(logit_2.shape[0])
            faceerr = faceerr.view(logit_3.shape[0])
            # loss    = criterion(logit, truth, False)
            lossdict  = {
                    "baseloss" : criterion(logit_0, truth),
                    "repalyloss" : criterion(logit_1, repalyerr),
                    "printloss" : criterion(logit_2, printerr),
                    "faceloss" : criterion(logit_3, faceerr),
                }
            loss = sum(lossdict.values())
            # print(lossdict["baseloss"].shape,loss.shape)
            correct, prob = metric(logit_0, truth)
            correct_repaly, prob_repaly = metric(logit_1, repalyerr)
            correct_print, prob_print = metric(logit_2, printerr)
            correct_face, prob_face = metric(logit_3, faceerr)
            # correctdict = {
            #         "baseacc" : correct,
            #         "repalyacc" : correct_repaly,
            #         "printacc" : correct_print,
            #         "facelacc" : correct_face,
            #     }
            # probdict = {
            #         "baseacer" : prob,
            #         "repalyacer" : prob_repaly,
            #         "printacer" : prob_print,
            #         "facelacer" : prob_face,
            #     }


        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        # correct_repalys.append(np.asarray(correct_repaly).reshape([1]))
        # correct_prints.append(np.asarray(correct_print).reshape([1]))
        # correct_faces.append(np.asarray(correct_face).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        prob_repalys.append(prob_repaly.data.cpu().numpy())
        prob_prints.append(prob_print.data.cpu().numpy())
        prob_faces.append(prob_face.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())
        repalylabels.append(repalyerr.data.cpu().numpy())
        printlabels.append(printerr.data.cpu().numpy())
        facelabels.append(faceerr.data.cpu().numpy())

    correct = np.concatenate(corrects)
    # loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    prob_repalys = np.concatenate(prob_repalys)
    prob_prints = np.concatenate(prob_prints)
    prob_faces = np.concatenate(prob_faces)
    labels = np.concatenate(labels)
    repalylabels = np.concatenate(repalylabels)
    printlabels = np.concatenate(printlabels)
    facelabels = np.concatenate(facelabels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    tpr, fpr, acc_repalys = calculate_accuracy(0.5, prob_repalys[:,1], repalylabels)
    tpr, fpr, acc_prints = calculate_accuracy(0.5, prob_prints[:,1], printlabels)
    tpr, fpr, acc_faces = calculate_accuracy(0.5, prob_faces[:,1], facelabels)
    accdict = {
        "acc" : acc,
        "acc_repalys" : acc_repalys,
        "acc_prints" : acc_prints,
        "acc_faces" : acc_faces
    }
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)
    acer_repalys,_,_,_,_ = ACER(0.5, prob_repalys[:, 1], repalylabels)
    acer_prints,_,_,_,_ = ACER(0.5, prob_prints[:, 1], printlabels)
    acer_faces,_,_,_,_ = ACER(0.5, prob_faces[:, 1], facelabels)
    acerdict = {
        "acer" : acer,
        "acer_repalys" : acer_repalys,
        "acer_prints" : acer_prints,
        "acer_faces" : acer_faces
    }

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels], accdict, acerdict, lossdict

def infer_test( net, test_loader):
    valid_num  = 0
    probs0 = []
    probs1 = []
    probs2 = []
    probs3 = []

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)
        input = input.cuda()

        with torch.no_grad():
            # logit,_,_   = net(input)
            # logit   = net(input)
            # logit = logit.view(b,n,2)
            # logit = torch.mean(logit, dim = 1, keepdim = False)
            # prob = F.softmax(logit, 1)
            logit = net(input)
            logit_0 = logit[0].cpu()
            logit_1 = logit[1].cpu()
            logit_2 = logit[2].cpu()
            logit_3 = logit[3].cpu()
            logit_0 = logit_0.view(b,n,2)
            logit_1 = logit_1.view(b,n,2)
            logit_2 = logit_2.view(b,n,2)
            logit_3 = logit_3.view(b,n,2)
            logit_0 = torch.mean(logit_0, dim = 1, keepdim = False)
            logit_1 = torch.mean(logit_1, dim = 1, keepdim = False)
            logit_2 = torch.mean(logit_2, dim = 1, keepdim = False)
            logit_3 = torch.mean(logit_3, dim = 1, keepdim = False)
            prob0 = F.softmax(logit_0, 1)
            prob1 = F.softmax(logit_1, 1)
            prob2 = F.softmax(logit_2, 1)
            prob3 = F.softmax(logit_3, 1)


        valid_num += len(input)
        probs0.append(prob0.data.cpu().numpy())
        probs1.append(prob1.data.cpu().numpy())
        probs2.append(prob2.data.cpu().numpy())
        probs3.append(prob3.data.cpu().numpy())

    probs0 = np.concatenate(probs0)
    probs1 = np.concatenate(probs1)
    probs2 = np.concatenate(probs2)
    probs3 = np.concatenate(probs3)
    return [probs0[:, 1],probs1[:, 1],probs2[:, 1],probs3[:, 1]]



