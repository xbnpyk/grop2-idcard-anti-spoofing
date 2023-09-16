import os
import time
import sys
import argparse
from process.data import FDDataset
from process.augmentation import get_augment
from process.data_helper import submission
from metric import metric, do_valid_test, infer_test
from model import get_model
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import *
from process.logging import Logger_tensorboard
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


#CUDA_VISIBLE_DEVICES=0 python train.py --model=ConvMixer --image_mode=color --image_size=96

def run_train(config):
    model_name = f'{config.model}_{config.image_mode}_{config.image_size}'
    if 'FaceBagNet' not in config.model:
        model_name += f'_{config.patch_size}_multjobs_adaptpretrain_onlyface'
    config.save_dir = os.path.join(config.save_dir, model_name)

    initial_checkpoint = config.pretrained_model
    criterion  = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(config.save_dir +'/checkpoint'):
        os.makedirs(config.save_dir +'/checkpoint')
    if not os.path.exists(config.save_dir +'/backup'):
        os.makedirs(config.save_dir +'/backup')
    if not os.path.exists(config.save_dir +'/backup'):
        os.makedirs(config.save_dir +'/backup')

    log = Logger()
    log_t = Logger_tensorboard()
    log_t.initialize(os.path.join(config.save_dir +'/backup'), 60)
    log.open(os.path.join(config.save_dir,model_name+'.txt'),mode='a')
    log.write('\tconfig.save_dir      = %s\n' % config.save_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    augment = get_augment(config.image_mode)
    train_dataset = FDDataset(mode = 'train', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment, dataroot="data/CASIA-SURF/phase1")
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.batch_size,
                                drop_last   = True,
                                num_workers = config.num_workers)

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment, dataroot="data/CASIA-SURF/phase1")

    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size = config.batch_size // 36,
                                drop_last  = False,
                                num_workers = config.num_workers)

    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('train_datasetsize : \n%s\n'%(len(train_dataset)))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_model(model_name=config.model, image_size=config.image_size, patch_size=config.patch_size)
    print(net)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(config.save_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage),strict=False)

    log.write('%s\n'%(type(net)))
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('model_name   lr   iter  epoch     |     loss      acer      acc    |     loss              acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    iter = 0
    i    = 0

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)

    start = timer()
    #-----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, weight_decay=0.0005)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          take_snapshot=False,
                                          out_dir=None,
                                          eta_min=1e-3)
    # sgdr = CosineAnnealingLR(optimizer,
    #                                       T_max=config.cycle_inter,
    #                                       eta_min=1e-3)

    global_min_acer = 1.0
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0
        start_time = time.time()

        for epoch in range(0, config.cycle_inter):
            sgdr.step()
            # scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            sum_train_loss = np.zeros(6,np.float32)
            sumstep = 0
            optimizer.zero_grad()

            for input, truth, repalyerr, printerr, faceerr in train_loader:
                datatime = time.time()
                iter = i + start_iter

                # one iteration update  -------------
                net.train()
                input = input.cuda()
                truth = truth.cuda()
                repalyerr = repalyerr.cuda()
                printerr = printerr.cuda()
                faceerr = faceerr.cuda()

                logit = net.forward(input)
                truth = truth.view(logit[0].shape[0])
                repalyerr = repalyerr.view(logit[1].shape[0])
                printerr = printerr.view(logit[2].shape[0])
                faceerr = faceerr.view(logit[3].shape[0])

                losses  = {
                    "baseloss" : criterion(logit[0], truth),
                    "repalyloss" : criterion(logit[1], repalyerr),
                    "printloss" : criterion(logit[2], printerr),
                    "faceloss" : criterion(logit[3], faceerr),
                }
                loss = sum(losses.values())
                precision,_ = metric(logit[0], truth)

                precisiones = {
                    "baseacc" : precision,
                    "repalyacc" : metric(logit[1], repalyerr)[0],
                    "printacc" : metric(logit[2], printerr)[0],
                    "facelacc" : metric(logit[3], faceerr)[0],
                }

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array(( loss.item(), precision.item(),))

                sumstep += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sumstep
                    sumstep = 0
                i=i+1
            log_t.training(iter, losses, precisiones,lr,time.time() - start_time, time.time() - datatime)

            if epoch >= config.cycle_inter // 2:
                net.eval()
                valid_loss,_ ,accs,acers,lossdict= do_valid_test(net, valid_loader, criterion)
                net.train()

                if valid_loss[1] < min_acer and epoch > 0:
                    min_acer = valid_loss[1]
                    ckpt_name = config.save_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')

                if valid_loss[1] < global_min_acer and epoch > 0:
                    global_min_acer = valid_loss[1]
                    ckpt_name = config.save_dir + '/checkpoint/global_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save global min acer model: ' + str(min_acer) + '\n')
                log_t.evaluation(iter, lossdict, acers, accs, lr,time.time() - start_time,time.time() - datatime)

            asterisk = ' '
            log.write(model_name+' Cycle %d: %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
                batch_loss[0], batch_loss[1],
                time_to_str((timer() - start), 'min')))
            start_time = time.time()

        ckpt_name = config.save_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')

def run_test(config, dir):
    config.save_dir = './Models'
    model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)+"_16_multjobs_new0.1"
    config.save_dir = os.path.join(config.save_dir, model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    ## net ---------------------------------------
    #net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = get_model(model_name=config.model, num_class=2, image_size=config.image_size, patch_size=config.patch_size)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()
    total_params = sum(p.numel() for p in net.parameters())
    total_params += sum(p.numel() for p in net.buffers())
    print(f'{total_params:,} total parameters.')

    if initial_checkpoint is not None:
        save_dir = os.path.join(config.save_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(config.save_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        # print(net.state_dict())
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(config.save_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(config.save_dir + '/checkpoint', dir))


    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers = config.num_workers)

    test_dataset = FDDataset(mode = 'test', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    test_loader  = DataLoader( test_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers = config.num_workers)

    criterion = softmax_cross_entropy_criterion
    net.eval()

    valid_loss,_ ,accs,acers,lossdict= do_valid_test(net, valid_loader, criterion)
    print('%0.6f  %0.6f  %0.3f  (%0.3f) acc:  %0.3f  %0.3f  %0.3f  %0.3f \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],accs["acc"],accs["acc_repalys"],accs["acc_prints"],accs["acc_faces"]))

    print('infer!!!!!!!!!')
    out = infer_test(net, test_loader)
    print('done')
    submission(out,accs,save_dir+'_noTTA_out.txt', mode='test')

def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA_out')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)

    parser.add_argument('--model', type=str, default='FaceBagNet')
    parser.add_argument('--image_mode', type=str, default = 'ir')
    parser.add_argument('--image_size', type=int, default = 96)
    parser.add_argument('--patch_size', type=int, default = 16)

    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--cycle_num', type=int, default=50)
    parser.add_argument('--cycle_inter', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./Models')

    config = parser.parse_args()
    print(config)
    main(config)
