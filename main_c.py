import os
import shutil
import argparse
from utils.config import cfg
from utils.tools.logger import Logger as Log
from tasks.main import main_worker
from utils.tools.file import mkdir_safe
import torch 

parser = argparse.ArgumentParser()

parser.add_argument('--train', default=True, action='store_true', help='command for train')
parser.add_argument('--test', default=False, action='store_true', help='command for test')
parser.add_argument('--config', default='/home/qbl/ccta/torchseg_ctn/tasks/configs/ccta_vessel.yaml', help='configure file path')
parser.add_argument('--resume', help='resume from latest checkpoint pth') #default=True,
parser.add_argument('--gpu', nargs='+', type=int, default=[5], help='which gpu to select')
parser.add_argument('--check_point', default=None, help='the check point path to store')


# parser.add_argument(
#        "opts",
#        help="Modify model config options using the command-line",
#        default=None,
#        nargs=argparse.REMAINDER,
#    )

def main():
    args = parser.parse_args()
    print(args)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['TRAIN.RESUME', args.resume])
    # torch.backends.cudnn.enabled = False

    if args.train:
        cfg.TASK.STATUS = 'train'
        cfg.OUTPUT_DIR = os.path.join('./results', cfg.TASK.NAME, 'train', cfg.OUTPUT_DIR)
        mkdir_safe(cfg.OUTPUT_DIR)

        Log.init(logfile_level='info',
                 log_file=os.path.join(cfg.OUTPUT_DIR, 'logger.log'),
                 stdout_level='info')

        if args.config != os.path.join(cfg.OUTPUT_DIR, os.path.basename(args.config)):
            shutil.copyfile(args.config, os.path.join(cfg.OUTPUT_DIR, os.path.basename(args.config)))

        cfg.TRAIN.RESUME = args.resume

    elif args.test:
        cfg.TASK.STATUS = 'test'
        cfg.TEST.SAVE_DIR = os.path.join('./results_train', cfg.TASK.NAME, 'test', cfg.TEST.SAVE_DIR)
        mkdir_safe(cfg.TEST.SAVE_DIR)
        print(cfg.TEST.SAVE_DIR)
        Log.init(logfile_level='info',
                 log_file=os.path.join(cfg.TEST.SAVE_DIR, 'logger.log'),
                 stdout_level='info')

        ct_pth = cfg.TEST.MODEL_PTH
        if args.check_point:
            ct_pth = args.check_point
        elif os.path.exists(os.path.join('./results', cfg.TASK.NAME, 'train', cfg.OUTPUT_DIR, ct_pth)):
            ct_pth = os.path.join('./results', cfg.TASK.NAME, 'train', cfg.OUTPUT_DIR, ct_pth)

        cfg.TEST.MODEL_PTH = ct_pth

    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in args.gpu)
    print(cfg)
    main_worker(args)  ### tasks.main


if __name__ == '__main__':
    main()
