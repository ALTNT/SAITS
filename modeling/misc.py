# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
import math

# Use math.inf instead of deprecated torch._six.inf
inf = math.inf



def init_distributed_mode(args):
    if args.dist_on_itp:#False
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:#False
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:#False
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        # setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):#在分布式训练中只让主进程（master process）输出日志信息 ，避免多个进程同时打印造成日志混乱
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print# 保存原始的print函数

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)# 检查是否强制打印
        force = force or (get_world_size() > 8)# 当进程数>8时也强制打印
        if is_master or force:# 只有主进程或强制打印时才执行
            now = datetime.datetime.now().time()# 添加时间戳
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)# 执行实际打印

    builtins.print = print# 替换全局的print函数



def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, use_amp=False):
        self.use_amp = use_amp
        if use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        #这里有两个语句
        #缩放损失并反向传播 ---将损失乘以大的缩放因子（如65536），防止梯度在FP16精度下变成0，防止梯度下溢  原理 ：FP16的最小正数约为6e-8，如果梯度小于这个值就会下溢为0
        # 接着执行反向传播计算梯度
        # 检查损失是否正常
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss}, skipping backward pass")
            return None
        
        if self.use_amp and self._scaler is not None:
            # 使用混合精度训练
            scaled_loss = self._scaler.scale(loss)
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                print(f"Warning: Scaled loss is {scaled_loss}, reducing scale factor")
                return None
            scaled_loss.backward(create_graph=create_graph)
        else:
            # 不使用混合精度训练，直接反向传播
            loss.backward(create_graph=create_graph)
        
        if update_grad:
            if self.use_amp and self._scaler is not None:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = get_grad_norm_(parameters)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                # 直接使用优化器
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters) if parameters else None
                optimizer.step()
        else:
            norm = None
        return norm

    def state_dict(self):#检查点（checkpoint）保存
        return self._scaler.state_dict()#主要包括当前的缩放因子、增长因子、回退计数器等AMP相关参数

    def load_state_dict(self, state_dict):#检查点（checkpoint）加载 #在检查点（checkpoint）加载时会调用此函数
        self._scaler.load_state_dict(state_dict)

def resume_model(config, model_without_ddp, optimizer, loss_scaler, final_output_dir):
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     print("Resume checkpoint %s" % args.resume)
    #     if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         args.start_epoch = checkpoint['epoch'] + 1
    #         if 'scaler' in checkpoint:
    #             loss_scaler.load_state_dict(checkpoint['scaler'])
    #         print("With optim & sched!")
    if config.TRAIN.RESUME :
        if config.TRAIN.CHECKPOINT == '':
            print(f"没有提供检查点文件，不进行恢复")
            raise ValueError("没有提供检查点文件，不进行恢复")
        if config.TRAIN.CHECKPOINT.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.TRAIN.CHECKPOINT, map_location='cpu', check_hash=True) 
        # model_state_file = os.path.join(final_output_dir, config.TRAIN.CHECKPOINT)
        model_state_file = config.TRAIN.CHECKPOINT
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
        last_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        model_without_ddp.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> 加载检查点 (epoch {checkpoint['epoch']})")
        best_model = True

        return last_epoch, best_perf, best_model
    if config.TRAIN.BEGIN_EPOCH> 0:
        return config.TRAIN.BEGIN_EPOCH, 0.0, False
    else:
        return 0, 0.0, False

# get_grad_norm_ 只计算范数，不修改梯度,而torch.nn.utils.clip_grad_norm_ 会修改梯度
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:#用于计算模型参数梯度的范数（norm），这是监控训练稳定性和调试的重要工具 默认L2范数
    if isinstance(parameters, torch.Tensor):#False
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:# 如果没有参数需要计算梯度范数，返回0
        return torch.tensor(0.)
    device = parameters[0].grad.device#device(type='cuda', index=0)
    if norm_type == inf:#False 无穷范数（L∞范数） 找到所有梯度中绝对值最大的元素 检测是否有异常大的梯度值
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:#True torch.stack([...]) 将所有参数的范数组成一个向量 p.grad.detach()从计算图中分离梯度张量
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)#tensor(3.5240, device='cuda:0')
    return total_norm

def adjust_learning_rate(optimizer, epoch, config, phase = 'pretrain'):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # if epoch < config.TRAIN.warmup_epochs:#args.warmup_epochs=40
    warmup_epochs = 15 if phase == 'pretrain' else 0
    epochs = config.TRAIN.END_EPOCH if phase == 'pretrain' else config.FINETUNE.END_EPOCH

    if epoch < warmup_epochs:
        lr = config.TRAIN.LR * epoch / warmup_epochs #0.00025*epoch/40#在前40个epoch中，学习率从0线性增长到目标学习率
    else:#Cosine Annealing（余弦退火）
        lr = config.TRAIN.min_lr + (config.TRAIN.LR - config.TRAIN.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    #这里学习率是通过直接修改optimizer的param_group的lr来实现的
    for param_group in optimizer.param_groups:#差异化学习率 例如：backbone可能使用较小的学习率，而新增的分类头使用较大的学习率
        if "lr_scale" in param_group:#好像只有微调的代码会进这里
            param_group["lr"] = lr * param_group["lr_scale"]#lr*0.003697205891018715
        else:
            param_group["lr"] = lr
    return lr

