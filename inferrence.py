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

class Inference():
    """用于执行模型推理过程的方法
    """
    def __init__(self, config):
        """模型网络初始化

        Args:
            config (argparse): 配置参数
        """
        self.config = config
        self.net = get_model(model_name=self.config.model, num_class=2, image_size=self.config.image_size, patch_size=self.config.patch_size)
        self.net = torch.nn.DataParallel(self.net)
        self.net =  self.net.cuda()
        self.modelloaded = False
    
    def load_checkpoint(self,checkpoint_name):
        """加载检查点参数

        Args:
            checkpoint_name : 需要加载的检查点名称
        """
        self.config.save_dir = './Models'
        self.config.save_dir = os.path.join(self.config.save_dir, checkpoint_name)
        initial_checkpoint = os.path.join(self.config.save_dir +'/checkpoint',r'global_min_acer_model.pth')
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        # print(net.state_dict())
        try:
            self.net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
            self.modelloaded = True
        except:
            print("未发现权重文件")
            self.modelloaded = False

    def run(self,input_dir,output_dir):
        """运行伪造预测程序
        该函数可自动遍历输入路径中的所以图片格式文件,并将预测结果保存至输出路径的“out.txt”文件中

        Args:
            input_dir: 输入数据路径
            output_dir: 图片保存路径
        """
        if self.modelloaded == False :
            print("请先加载模型参数文件")
            return
        self.outputpath = os.path.join(output_dir,"out.txt")#文件输出路径
        augment = get_augment(self.config.image_mode)
        inf_dataset = FDDataset(mode = 'test', modality=self.config.image_mode,image_size=self.config.image_size,
                                fold_index=self.config.train_fold_index,augment=augment,dataroot=input_dir)
        if len(inf_dataset) == 0:
            print("未找到图片文件")
            return
        inf_loader  = DataLoader( inf_dataset,
                                    shuffle=False,
                                    batch_size  = self.config.batch_size,
                                    drop_last   = False,
                                    num_workers = self.config.num_workers)
        self.net.eval()
        print('infer!!!!!!!!!')
        probs0 = []#存储真伪预测结果
        probs1 = []#存储打印预测结果
        probs2 = []#存储翻拍预测结果
        probs3 = []#存储贴照片预测结果

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
        submission(out,None,self.outputpath, mode='test')

def main(config):
    """推理阶段主函数
    具有模型初始化、模型参数加载、运行检测等功能

    Args:
        config (argparse) : 配置参数
    """
    # 初始化模型
    net = Inference(config)
    while True : #等待命令行输入 
        msg = input()
        #退出程序
        if msg == "exit":
            exit()
        # 加载模型参数
        elif len(msg.split(" "))==2 and msg.split(" ")[0]=="loadcheckpoint":
            net.load_checkpoint(msg.split(" ")[1])
        # 运行检测
        elif len(msg.split(" "))==2 and msg.split(" ")[0]=="datapath":
            net.run( msg.split(" ")[1], 'global_test_36_TTA_out')

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
