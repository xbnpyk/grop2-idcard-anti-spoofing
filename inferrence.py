import os
import argparse
from process.data import FDDataset
from process.augmentation import get_augment
from process.data_helper import submission
from model import get_model
from tqdm import tqdm
from utils import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


#CUDA_VISIBLE_DEVICES=0 python train.py --model=ConvMixer --image_mode=color --image_size=96
class Inference():
    def __init__(self, config):
        ## net ---------------------------------------
        #net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
        self.config = config
        self.net = get_model(model_name=self.config.model, num_class=2, image_size=self.config.image_size, patch_size=self.config.patch_size)
        self.net = torch.nn.DataParallel(self.net)
        self.net =  self.net.cuda()
        total_params = sum(p.numel() for p in self.net.parameters())
        total_params += sum(p.numel() for p in self.net.buffers())
        print(f'{total_params:,} total parameters.')
    
    def load_checkpoint(self,pretrained_model):
        self.config.save_dir = './Models'
        model_name = self.config.model + '_' + self.config.image_mode + '_' + str(self.config.image_size)+"_16_multjobs_new0.1"
        self.config.save_dir = os.path.join(self.config.save_dir, model_name)
        initial_checkpoint = pretrained_model
        if initial_checkpoint is not None:
            
            initial_checkpoint = os.path.join(self.config.save_dir +'/checkpoint',initial_checkpoint)
            print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
            # print(net.state_dict())
            try:
                self.net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
            except:
                print("未发现权重文件")

    def run_test(self,input_dir,output_dir):
        self.save_dir = os.path.join(self.config.save_dir + '/checkpoint', output_dir)
        augment = get_augment(self.config.image_mode)
        inf_dataset = FDDataset(mode = 'test', modality=self.config.image_mode,image_size=self.config.image_size,
                                fold_index=self.config.train_fold_index,augment=augment,dataroot=input_dir)
        inf_loader  = DataLoader( inf_dataset,
                                    shuffle=False,
                                    batch_size  = self.config.batch_size,
                                    drop_last   = False,
                                    num_workers = self.config.num_workers)
        self.net.eval()
        print('infer!!!!!!!!!')
        valid_num  = 0
        probs0 = []
        probs1 = []
        probs2 = []
        probs3 = []

        for i, (input, truth) in enumerate(tqdm(inf_loader)):
            b,n,c,w,h = input.size()
            input = input.view(b*n,c,w,h)
            input = input.cuda()

            with torch.no_grad():
                logit = self.net(input)
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
        out =  [probs0[:, 1],probs1[:, 1],probs2[:, 1],probs3[:, 1]]
        print('done')
        submission(out,None,self.save_dir+'_noTTA_out.txt', mode='test')

def main(config):
    #初始化模型
    net = Inference(config)
    #加载模型参数
    net.load_checkpoint(config.pretrained_model)
    #run
    net.run_test( None, 'global_test_36_TTA_out')

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
    config.pretrained_model = r'global_min_acer_model.pth'
    print(config)
    main(config)
