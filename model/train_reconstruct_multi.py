"""
Main script for image reconstruction experiments
Author: Patrick Ebel (github/PatrickTUM), based on the scripts of
        Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""


import os
import sys
import time
import json
import random
import pprint
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from parse_args import create_parser
from data.dataLoader import SEN12MSCR, SEN12MSCRTS
from src.model_utils import get_model, save_model, freeze_layers, load_model, load_checkpoint
from src.learning.metrics import img_metrics, avg_img_metrics
from torch.nn.parallel import DistributedDataParallel
import torch
import torchnet as tnt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src import utils, losses
from src.learning.weight_init import weight_init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


S2_BANDS = 13
parser = create_parser(mode='train')
# 同时注册 --local-rank 和 --local_rank，dest 都是 local_rank，默认 None
parser.add_argument(
    '--local-rank', '--local_rank',
    dest='local_rank', type=int, default=None,
    help='Local process rank for DDP (from CLI or env)'
)
parser.add_argument(
    '--dist', action='store_true',
    help='启用多卡 DDP'
)

# 1) 先解析所有参数
args = parser.parse_args()

# 2) 把 args 转成 config（保持你原来的 list_args 列表）
config = utils.str2list(
    args,
    list_args=["encoder_widths", "decoder_widths", "out_conv"]
)

# 3) 根据 CLI 或环境变量决定 local_rank
config.local_rank = (
    args.local_rank
    if args.local_rank is not None
    else int(os.environ.get('LOCAL_RANK', 0))
)

# 4) 设置当前进程的 GPU
torch.cuda.set_device(config.local_rank)
# Add DDP arguments to parser


if config.model in['unet', 'utae']:
    assert len(config.encoder_widths) == len(config.decoder_widths)
    config.loss = 'l2'
    if config.model=='unet':
        # train U-Net from scratch
        config.pretrain=True
        config.trained_checkp = ''

if config.pretrain:  # pre-training is on a single time point
    config.input_t = config.n_head = 1
    config.sample_type = 'pretrain'
    if config.model=='unet': config.batch_size = 32
    config.positional_encoding = False

if config.loss in ['GNLL', 'MGNLL']:
    # for univariate losses, default to univariate mode (batched across channels)
    if config.loss in ['GNLL']: config.covmode = 'uni' 

    if config.covmode == 'iso':
        config.out_conv[-1] += 1
    elif config.covmode in ['uni', 'diag']:
        config.out_conv[-1] += S2_BANDS
        config.var_nonLinearity = 'softplus'

# grab the PID so we can look it up in the logged config for server-side process management
config.pid = os.getpid()

# import & re-load a previous configuration, e.g. to resume training
if config.resume_from:
    load_conf = os.path.join(config.res_dir, config.experiment_name, 'conf.json')
    if config.experiment_name != config.trained_checkp.split('/')[-2]: 
        raise ValueError("Mismatch of loaded config file and checkpoints")
    with open(load_conf, 'rt') as f:
        t_args = argparse.Namespace()
        # do not overwrite the following flags by their respective values in the config file
        no_overwrite = ['pid', 'num_workers', 'root1', 'root2', 'root3', 'resume_from', 'trained_checkp', 'epochs', 'encoder_widths', 'decoder_widths', 'lr']
        conf_dict = {key:val for key,val in json.load(f).items() if key not in no_overwrite}
        for key, val in vars(config).items(): 
            if key in no_overwrite: conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
config = utils.str2list(config, list_args=["encoder_widths", "decoder_widths", "out_conv"])

# resume at a specified epoch and update optimizer accordingly
if config.resume_at >= 0:
    config.lr = config.lr * config.gamma**config.resume_at


# fix all RNG seeds,
# throw the whole bunch at 'em
def seed_packages(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# seed everything
seed_packages(config.rdm_seed)
# seed generators for train & val/test dataloaders
f, g = torch.Generator(), torch.Generator()
f.manual_seed(config.rdm_seed + 0)  # note:  this may get re-seeded each epoch
g.manual_seed(config.rdm_seed)      #        keep this one fixed

if __name__ == "__main__": pprint.pprint(config)

# instantiate tensorboard logger
rank = 0  # 临时设置，将在main函数中正确初始化
is_master = True  # 临时设置，将在main函数中正确初始化
writer = None  # 将在main函数中初始化
def plot_img(imgs, mod, plot_dir, file_id=None):
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    try:
        imgs = imgs.cpu().numpy()
        for tdx, img in enumerate(imgs): # iterate over temporal dimension
            time = '' if imgs.shape[0] == 1 else f'_t-{tdx}'
            if mod in ["pred", "in", "target", "s2"]:
                rgb = [3,2,1] if img.shape[0]==S2_BANDS else [5,4,3]
                img, val_min, val_max = img[rgb, ...], 0, 1
            elif mod == "s1":
                img, val_min, val_max = img[[0], ...], 0, 1
            elif mod == "mask":
                img, val_min, val_max = img[[0], ...], 0, 1
            elif mod == "err":
                img, val_min, val_max = img[[0], ...], 0, 0.01
            elif mod == "var":
                img, val_min, val_max = img[[0], ...], 0, 0.000025
            else: raise NotImplementedError
            if file_id is not None: # export into file name
                img = img.clip(val_min, val_max) # note: this only removes outliers, vmin/vmax below do the global rescaling (else doing instance-wise min/max scaling)
                plt.imsave(os.path.join(plot_dir, f'img-{file_id}_{mod}{time}.png'), np.moveaxis(img,0,-1).squeeze(), dpi=100, cmap='gray', vmin=val_min, vmax=val_max)
    except: 
        if isinstance(imgs, plt.Figure): # the passed argument is a pre-rendered figure
            plt.savefig(os.path.join(plot_dir, f'img-{file_id}_{mod}.png'), dpi=100)
        else: raise NotImplementedError


def export(arrs, mod, export_dir, file_id=None):
    if not os.path.exists(export_dir): os.makedirs(export_dir)
    for tdx, arr in enumerate(arrs): # iterate over temporal dimension
        num = '' if arrs.shape[0] == 1 else f'_t-{tdx}'
        np.save(os.path.join(export_dir, f'img-{file_id}_{mod}{num}.npy'), arr.cpu())

def prepare_data(batch, device, config):
    if config.pretrain: return prepare_data_mono(batch, device, config)
    else: return prepare_data_multi(batch, device, config)

def prepare_data_mono(batch, device, config):
    x = batch['input']['S2'].to(device).unsqueeze(1)
    if config.use_sar: 
        x = torch.cat((batch['input']['S1'].to(device).unsqueeze(1), x), dim=2)
    m = batch['input']['masks'].to(device).unsqueeze(1)
    y = batch['target']['S2'].to(device).unsqueeze(1)
    return x, y, m

def prepare_data_multi(batch, device, config):
    in_S2       = recursive_todevice(batch['input']['S2'], device)
    in_S2_td    = recursive_todevice(batch['input']['S2 TD'], device)
    if config.batch_size>1: in_S2_td = torch.stack((in_S2_td)).T
    in_m        = torch.stack(recursive_todevice(batch['input']['masks'], device)).swapaxes(0,1)
    target_S2   = recursive_todevice(batch['target']['S2'], device)
    y           = torch.cat(target_S2,dim=0).unsqueeze(1)

    if config.use_sar: 
        in_S1 = recursive_todevice(batch['input']['S1'], device)
        in_S1_td = recursive_todevice(batch['input']['S1 TD'], device)
        if config.batch_size>1: in_S1_td = torch.stack((in_S1_td)).T
        x     = torch.cat((torch.stack(in_S1,dim=1), torch.stack(in_S2,dim=1)),dim=2)
        dates = torch.stack((torch.tensor(in_S1_td),torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
    else:
        x     = torch.stack(in_S2,dim=1)
        dates = torch.tensor(in_S2_td).float().to(device)
    
    return x, y, in_m, dates


def log_aleatoric(writer, config, mode, step, var, name, img_meter=None):

    # if var is of shape [B x 1 x C x C x H x W] then it's a covariance tensor
    if len(var.shape) > 5: 
        covar = var
        # get [B x 1 x C x H x W] variance tensor
        var   = var.diagonal(dim1=2, dim2=3).moveaxis(-1,2)

        # compute spatial-average to visualize patch-wise covariance matrices
        patch_covmat = covar.mean(dim=-1).mean(dim=-1).squeeze(dim=1)
        for bdx, img in enumerate(patch_covmat): # iterate over [B x C x C] covmats
            img = img.detach().numpy()

            max_abs = max(abs(img.min()), abs(img.max()))
            scale_rel_left, scale_rel_right = -max_abs, +max_abs
            fig = continuous_matshow(img, min=scale_rel_left, max=scale_rel_right)
            writer.add_figure(f'Img/{mode}/patch covmat relative {bdx}',fig, step)
            scale_center0_absolute = 1/4 * 1**2 # assuming covmat has been rescaled already, this is an upper bound
            fig = continuous_matshow(img, min=-scale_center0_absolute, max=scale_center0_absolute)
            writer.add_figure(f'Img/{mode}/patch covmat absolute {bdx}',fig, step)

    # aleatoric uncertainty: comput during train, val and test
    # note: the quantile statistics are computed solely over the variances (and would be much different if involving covariances, e.g. in the isotopic case)
    avg_var     = torch.mean(var, dim=2, keepdim=True) # avg over bands, note: this only considers variances (else diag COV's avg would be tiny)
    q50         = avg_var[:,0,...].view(avg_var.shape[0],-1).median(dim=-1)[0].detach().clone()
    q75         = avg_var[:,0,...].view(avg_var.shape[0],-1).quantile(0.75,dim=-1).detach().clone()
    q50, q75    = q50[0], q75[0] # take batch's first item as a summary
    binning     = 256 # see: https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_histogram

    if config.loss in ["GNLL", 'MGNLL']:
        writer.add_image(f'Img/{mode}/{name}aleatoric [0,1]', avg_var[0,0,...].clip(0, 1), step, dataformats='CHW') # map image to [0, 1]
        writer.add_image(f'Img/{mode}/{name}aleatoric [0,q75]', avg_var[0,0,...].clip(0.0, q75)/q75, step, dataformats='CHW') # map image to [0, q75]
        writer.add_histogram(f'Hist/{mode}/{name}aleatoric', avg_var[0,0,...].flatten().clip(0,1), step, bins=binning, max_bins=binning)
    else: raise NotImplementedError

    writer.add_scalar(f'{mode}/{name}aleatoric median all', q50, step)
    writer.add_scalar(f'{mode}/{name}aleatoric q75 all', q75, step)
    if img_meter is not None: 
        writer.add_scalar(f'{mode}/{name}UCE SE', img_meter.value()['UCE SE'], step)
        writer.add_scalar(f'{mode}/{name}AUCE SE', img_meter.value()['AUCE SE'], step)


def log_train(writer, config, model, step, x, out, y, in_m, name='', var=None):
    # logged loss is before rescaling by learning rate
    _, loss = model.criterion, model.loss_G.cpu()
    if name != '': name = f'model_{name}/'     
    
    writer.add_scalar(f'train/{name}{config.loss}', loss, step)
    writer.add_scalar(f'train/{name}total', loss, step)
    # use add_images for batch-wise adding across temporal dimension
    if config.use_sar:
        writer.add_image(f'Img/train/{name}in_s1', x[0,:,[0], ...], step, dataformats='NCHW')
        writer.add_image(f'Img/train/{name}in_s2', x[0,:,[5,4,3], ...], step, dataformats='NCHW')
    else:
        writer.add_image(f'Img/train/{name}in_s2', x[0,:,[3,2,1], ...], step, dataformats='NCHW')
    writer.add_image(f'Img/train/{name}out', out[0,0,[3,2,1], ...], step, dataformats='CHW')
    writer.add_image(f'Img/train/{name}y', y[0,0,[3,2,1], ...], step, dataformats='CHW')
    writer.add_image(f'Img/train/{name}m', in_m[0,:,None, ...], step, dataformats='NCHW')

    # analyse cloud coverage

    # covered at ALL time points (AND) or covered at ANY time points (OR)
    #and_m, or_m = torch.prod(in_m[0,:, ...], dim=0, keepdim=True), torch.sum(in_m[0,:, ...], dim=0, keepdim=True).clip(0,1)
    and_m, or_m = torch.prod(in_m, dim=1, keepdim=True), torch.sum(in_m, dim=1, keepdim=True).clip(0,1)
    writer.add_scalar(f'train/{name}OR m %', or_m.float().mean(), step)
    writer.add_scalar(f'train/{name}AND m %', and_m.float().mean(), step)
    writer.add_image(f'Img/train/{name}AND m', and_m, step, dataformats='NCHW')
    writer.add_image(f'Img/train/{name}OR m',  or_m, step, dataformats='NCHW')

    and_m_gray = in_m.float().mean(axis=1).cpu()
    for bdx, img in enumerate(and_m_gray):
        fig = discrete_matshow(img, n_colors=config.input_t)
        writer.add_figure(f'Img/train/temp overlay m {bdx}',fig, step)

    if var is not None:  
        # log aleatoric uncertainty statistics, excluding computation of ECE
        log_aleatoric(writer, config, 'train', step, var, name, img_meter=None)

def discrete_matshow(data, n_colors=5, min=0, max=1):
    fig, ax = plt.subplots()
    # get discrete colormap
    cmap = plt.get_cmap('gray', n_colors+1)
    ax.matshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.axis('off')
    fig.tight_layout()
    return fig

def continuous_matshow(data, min=0, max=1):
    fig, ax = plt.subplots()
    # get discrete colormap
    cmap = plt.get_cmap('seismic')
    ax.matshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.axis('off')
    # optionally: provide a colorbar and tick at integers
    # cax = plt.colorbar(mat, ticks=np.arange(min, max + 1))
    return fig

def iterate(model, data_loader, config, writer, mode="train", epoch=None, device=None):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if len(data_loader) == 0: raise ValueError("Received data loader with zero samples!")
    # loss meter, needs 1 meter per scalar (see https://tnt.readthedocs.io/en/latest/_modules/torchnet/meter/averagevaluemeter.html);
    loss_meter = tnt.meter.AverageValueMeter()
    img_meter  = avg_img_metrics()

    # collect sample-averaged uncertainties and errors
    errs, errs_se, errs_ae,  vars_aleatoric= [], [], [], []

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        step = (epoch-1)*len(data_loader)+i

        if config.sample_type == 'cloudy_cloudfree':
            x, y, in_m, dates = prepare_data(batch, device, config)
        elif config.sample_type == 'pretrain':
            x, y, in_m = prepare_data(batch, device, config)
            dates = None
        else:
            raise NotImplementedError
        inputs = {'A': x, 'B': y, 'dates': dates, 'masks': in_m}


        if mode != "train": # val or test
            with torch.no_grad():
                # compute single-model mean and variance predictions
                model.set_input(inputs)
                model.forward()
                model.get_loss_G()
                model.rescale()
                out = model.fake_B
                if hasattr(model.netG, 'variance') and model.netG.variance is not None:
                    var = model.netG.variance
                    model.netG.variance = None
                else:
                    var = out[:, :, S2_BANDS:, ...]
                out = out[:, :, :S2_BANDS, ...]
                batch_size = y.size()[0]

                for bdx in range(batch_size):
                    # only compute statistics on variance estimates if using e.g. NLL loss or combinations thereof
                    
                    if config.loss in ['GNLL', 'MGNLL']:
                        
                        # if the variance variable is of shape [B x 1 x C x C x H x W] then it's a covariance tensor
                        if len(var.shape) > 5: 
                            covar = var
                            # get [B x 1 x C x H x W] variance tensor
                            var   = var.diagonal(dim1=2, dim2=3).moveaxis(-1,2)

                        extended_metrics = img_metrics(y[bdx], out[bdx], var=var[bdx])
                        vars_aleatoric.append(extended_metrics['mean var']) 
                        errs.append(extended_metrics['error'])
                        errs_se.append(extended_metrics['mean se'])
                        errs_ae.append(extended_metrics['mean ae'])
                    else:
                        extended_metrics = img_metrics(y[bdx], out[bdx])
                    
                    img_meter.add(extended_metrics)
                    idx = (i*batch_size+bdx) # plot and export every k-th item
                    if config.plot_every>0 and idx % config.plot_every == 0:
                        plot_dir = os.path.join(config.res_dir, config.experiment_name, 'plots', f'epoch_{epoch}', f'{mode}')
                        plot_img(x[bdx], 'in', plot_dir, file_id=idx)
                        plot_img(out[bdx], 'pred', plot_dir, file_id=idx)
                        plot_img(y[bdx], 'target', plot_dir, file_id=idx)
                        plot_img(((out[bdx]-y[bdx])**2).mean(1, keepdims=True), 'err', plot_dir, file_id=idx)
                        plot_img(discrete_matshow(in_m.float().mean(axis=1).cpu()[bdx], n_colors=config.input_t), 'mask', plot_dir, file_id=idx)
                        if var is not None: plot_img(var.mean(2, keepdims=True)[bdx], 'var', plot_dir, file_id=idx)
                    if config.export_every>0 and idx % config.export_every == 0:
                        export_dir = os.path.join(config.res_dir, config.experiment_name, 'export', f'epoch_{epoch}', f'{mode}')
                        export(out[bdx], 'pred', export_dir, file_id=idx)
                        export(y[bdx], 'target', export_dir, file_id=idx)
                        if var is not None: 
                            try: export(covar[bdx], 'covar', export_dir, file_id=idx)
                            except: export(var[bdx], 'var', export_dir, file_id=idx)
        else: # training
            
            # compute single-model mean and variance predictions
            model.set_input(inputs)
            model.optimize_parameters()
            step = (epoch - 1) * len(data_loader) + i # not using model.forward() directly
            if step % config.display_step == 0:
                branch_modules = {
                        'uncrtaints': 'netG.un_model',
                        'mamba':      'netG.mamba_model',
                        'fusion':     'netG.scale_net',
                    }
                for tag, module_name in branch_modules.items():
                    sq_sum = 0.0
                    for name, p in model.named_parameters():
            # DDP 模    式下 name 可能以 "module." 开头，只要包含子模块关键字即算
                        if module_name in name and p.grad is not None:
                            sq_sum += p.grad.data.norm(2).item() ** 2
                    grad_norm = sq_sum ** 0.5
                    print(f"[GradNorm] Epoch {epoch} Step {step} | {tag:10s}: {grad_norm:.4f}")
                    writer.add_scalar(f'train/grad_norm_{tag}', grad_norm, step)
            # —— 分支监控结束 —— #

            out    = model.fake_B.detach().cpu()

            # read variance predictions stored on generator
            if hasattr(model.netG, 'variance') and model.netG.variance is not None:
                var = model.netG.variance.cpu()
            else:
                var = out[:, :, S2_BANDS:, ...]
            out = out[:, :, :S2_BANDS, ...]

            if config.plot_every>0:
                plot_out = out.detach().clone()
                batch_size = y.size()[0]
                for bdx in range(batch_size):
                    idx = (i*batch_size+bdx) # plot and export every k-th item
                    if idx % config.plot_every == 0:
                        plot_dir = os.path.join(config.res_dir, config.experiment_name, 'plots', f'epoch_{epoch}', f'{mode}')
                        plot_img(x[bdx], 'in', plot_dir, file_id=i)
                        plot_img(plot_out[bdx], 'pred', plot_dir, file_id=i)
                        plot_img(y[bdx], 'target', plot_dir, file_id=i)

        if mode == "train":
            # periodically log stats
            if step%config.display_step==0:
                out, x, y, in_m = out.cpu(), x.cpu(), y.cpu(), in_m.cpu()
                if config.loss in ['GNLL', 'MGNLL']:
                    var = var.cpu()
                    log_train(writer, config, model, step, x, out, y, in_m, var=var)
                else:
                    log_train(writer, config, model, step, x, out, y, in_m)
                if hasattr(model, 'aux_info'):
                    aux = model.aux_info
                    print(f"[AUX INFO] Epoch {epoch}, Step {step} | "
                        f"loss_un: {aux['loss_un']:.4f}, "
                        f"loss_mamba: {aux['loss_mamba']:.4f}, "
                        f"loss_fused: {aux['loss_fused']:.4f}")

                    writer.add_scalar('train/loss_un',     aux['loss_un'], step)
                    writer.add_scalar('train/loss_mamba',  aux['loss_mamba'], step)
                    writer.add_scalar('train/loss_fused',  aux['loss_fused'], step)
        # log the loss, computed via model.backward_G() at train time & via model.get_loss_G() at val/test time
        loss_meter.add(model.loss_G.item())

        # after each batch, close any leftover figures
        plt.close('all')

    # --- end of epoch ---
    # after each epoch, log the loss metrics
    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {f"{mode}_epoch_time": total_time}
    # log the loss, only computed within model.backward_G() at train time
    metrics[f"{mode}_loss"] = loss_meter.value()[0]

    if mode == "train": # after each epoch, update lr acc. to scheduler
        current_lr = model.optimizer_G.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('Etc/train/lr', current_lr, step)
        model.scheduler_G.step()

    if mode == "test" or mode == "val":
        # log the metrics

        # log image metrics
        for key, val in img_meter.value().items(): writer.add_scalar(f'{mode}/{key}', val, step)

        # any loss is currently only computed within model.backward_G() at train time
        writer.add_scalar(f'{mode}/loss', metrics[f"{mode}_loss"], step)

        # use add_images for batch-wise adding across temporal dimension
        if config.use_sar:
            writer.add_image(f'Img/{mode}/in_s1', x[0,:,[0], ...], step, dataformats='NCHW')
            writer.add_image(f'Img/{mode}/in_s2', x[0,:,[5,4,3], ...], step, dataformats='NCHW')
        else:
            writer.add_image(f'Img/{mode}/in_s2', x[0,:,[3,2,1], ...], step, dataformats='NCHW')
        writer.add_image(f'Img/{mode}/out', out[0,0,[3,2,1], ...], step, dataformats='CHW')
        writer.add_image(f'Img/{mode}/y', y[0,0,[3,2,1], ...], step, dataformats='CHW')
        writer.add_image(f'Img/{mode}/m', in_m[0,:,None, ...], step, dataformats='NCHW')


        # compute Expected Calibration Error (ECE)
        if config.loss in ['GNLL', 'MGNLL']:
            sorted_errors_se   = compute_ece(vars_aleatoric, errs_se, len(data_loader.dataset), percent=5)
            sorted_errors      = {'se_sortAleatoric': sorted_errors_se}
            plot_discard(sorted_errors['se_sortAleatoric'], config, mode, step, is_se=True)

            # compute ECE 
            uce_l2, auce_l2 = compute_uce_auce(vars_aleatoric, errs, len(data_loader.dataset), percent=5, l2=True, mode=mode, step=step)

            # no need for a running mean here
            img_meter.value()['UCE SE']  = uce_l2.cpu().numpy().item()
            img_meter.value()['AUCE SE'] = auce_l2.cpu().numpy().item()

        if config.loss in ['GNLL', 'MGNLL']:
            log_aleatoric(writer, config, mode, step, var,  f'model/', img_meter)

        return metrics, img_meter.value()
    else:
        return metrics

def plot_discard(sorted_errors, config, mode, step, is_se=True):
    metric = 'SE' if is_se else 'AE'

    fig, ax = plt.subplots()
    x_axis  = np.arange(0.0, 1.0, 0.05)
    ax.scatter(x_axis, sorted_errors, c="b", alpha=1.0, marker=r'.', label=f"{metric}, sorted by uncertainty")

    # fit a linear regressor with slope b and intercept a
    sorted_errors[np.isnan(sorted_errors)] = np.nanmean(sorted_errors)
    b, a  = np.polyfit(x_axis, sorted_errors, deg=1)
    x_seq = np.linspace(0, 1.0, num=1000)
    ax.plot(x_seq, a + b * x_seq, c="k", lw=1.5, alpha=0.75, label=f"linear fit, {round(a, 3)} + {round(b, 3)} * x")
    plt.xlabel("Fraction of samples, sorted ascendingly by uncertainty")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    plt.grid()
    fig.tight_layout()
    writer.add_figure(f'Img/{mode}/discard_uncertain',fig, step)
    if mode=='test': # export the final test split plots for print
        path_to = os.path.join(config.res_dir, config.experiment_name)
        print(f'Logging discard plots to path {path_to}')
        fig.savefig(os.path.join(path_to, f'plot_{mode}_{metric}_discard.png'), bbox_inches='tight', dpi=int(1e3))
        fig.savefig(os.path.join(path_to, f'plot_{mode}_{metric}_discard.pdf'), bbox_inches='tight', dpi=int(1e3))


def compute_ece(vars, errors, n_samples, percent=5):
    # rank sample-averaged uncertainties ascendingly, and errors accordingly
    _, vars_indices = torch.sort(torch.Tensor(vars))
    errors = torch.Tensor(errors)
    errs_sort = errors[vars_indices]
    # incrementally remove 5% of errors, ranked by highest uncertainty
    bins = torch.linspace(0, n_samples, 100//percent+1, dtype=int)[1:]
    # get uncertainty-sorted cumulative errors, i.e. at x-tick 65% we report the average error for the 65% most certain predictions
    sorted_errors = np.array([torch.nanmean(errs_sort[:rdx]).cpu().numpy() for rdx in bins])

    return sorted_errors 


binarize   = lambda arg, n_bins, floor=0, ceil=1: np.digitize(arg, bins=np.linspace(floor, ceil, num=n_bins)[1:])

def compute_uce_auce(var, errors, n_samples, percent=5, l2=True, mode='val', step=0):
    n_bins = 100//percent
    var, errors = torch.Tensor(var), torch.Tensor(errors)

    # metric: IN:  standard deviation & error
    #         OUT: either root mean variance & root mean squared error or mean standard deviation & mean absolute error
    metric = lambda arg: torch.sqrt(torch.mean(arg**2)) if l2 else torch.mean(torch.abs(arg))
    m_str  = 'L2' if l2 else 'L1'

    # group uncertainty values into n_bins 
    var_idx = torch.Tensor(binarize(var, n_bins, floor=var.min(), ceil=var.max()))

    # compute bin-wise statistics, defaults to nan if no data contained in bin
    bk_var, bk_err = torch.empty(n_bins), torch.empty(n_bins)
    for bin_idx in range(n_bins): # for each of the n_bins ... 
        bk_var[bin_idx] = metric(var[var_idx==bin_idx].sqrt())  # note: taking the sqrt to wrap into metric function,
        bk_err[bin_idx] = metric(errors[var_idx==bin_idx])      # apply same metric function on error

    calib_err = torch.abs(bk_err-bk_var)                        # calibration error: discrepancy of error vs uncertainty
    bk_weight = torch.histogram(var_idx, n_bins)[0]/n_samples   # fraction of total data per bin, for bin-weighting
    uce  = torch.nansum(bk_weight * calib_err)                  # calc. weighted UCE, 
    auce = torch.nanmean(calib_err)                             # calc. unweighted AUCE

    # plot bin-wise error versus bin-wise uncertainty
    fig, ax = plt.subplots()
    x_min, x_max = bk_var[~bk_var.isnan()].min(), bk_var[~bk_var.isnan()].max()
    y_min, y_max = 0, bk_err[~bk_err.isnan()].max()
    x_axis  = np.linspace(x_min, x_max, num=n_bins)

    ax.plot(x_axis, x_axis)                                     # diagonal reference line
    ax.bar(x_axis, bk_err, width=x_axis[1]-x_axis[0], alpha=0.75, edgecolor='k', color='gray')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Uncertainty")
    plt.ylabel(f"{m_str} Error")
    plt.legend(loc='upper left')
    plt.grid()
    fig.tight_layout()
    writer.add_figure(f'Img/{mode}/err_vs_var_{m_str}',fig, step)
    
    return uce, auce


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(os.path.join(config.res_dir, config.experiment_name), exist_ok=True)

def checkpoint(log, config):
    with open(
        os.path.join(config.res_dir, config.experiment_name, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)

def save_results(metrics, path, split='test'):
    with open(
        os.path.join(path, f"{split}_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)


# check for file of pre-computed statistics, e.g. indices or cloud coverage
def import_from_path(split, config):
    if os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'util', 'precomputed')):
        import_path = os.path.join(os.path.dirname(os.getcwd()), 'util', 'precomputed', f'generic_{config.input_t}_{split}_{config.region}_s2cloudless_mask.npy')
    else:
        import_path = os.path.join(config.precomputed, f'generic_{config.input_t}_{split}_{config.region}_s2cloudless_mask.npy')
    import_data_path = import_path if os.path.isfile(import_path) else None
    return import_data_path
    

def main(config):
    global rank, is_master, writer  # 添加这行
    
    if config.dist:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(config.local_rank)
    
    # 在DDP初始化后正确设置rank和writer
    rank = dist.get_rank() if config.dist else 0
    is_master = (not config.dist) or (rank == 0)
    
    log_dir = os.path.join(os.path.dirname(config.res_dir),
                           "logs", config.experiment_name)
    
    if is_master:
        writer = SummaryWriter(log_dir)
    else:
        class _Dummy:
            def add_scalar(*a, **k):     pass
            def add_image(*a, **k):      pass
            def add_histogram(*a, **k):  pass
            def add_figure(*a, **k):     pass
            def close(*a, **k):          pass
        writer = _Dummy()


    # define data sets
    if config.pretrain: # pretrain / training on mono-temporal data
        dt_train    = SEN12MSCR(os.path.expanduser(config.root3), split='train', region=config.region, sample_type=config.sample_type)
        dt_val      = SEN12MSCR(os.path.expanduser(config.root3), split='val', region=config.region, sample_type=config.sample_type) 
        dt_test     = SEN12MSCR(os.path.expanduser(config.root3), split='test', region=config.region, sample_type=config.sample_type)
    else:
        dt_train    = SEN12MSCRTS(os.path.expanduser(config.root1), split='train', region=config.region, sample_type=config.sample_type, sampler = 'random' if config.vary_samples else 'fixed', n_input_samples=config.input_t, import_data_path=import_from_path('train', config), min_cov=config.min_cov, max_cov=config.max_cov)
        dt_val      = SEN12MSCRTS(os.path.expanduser(config.root2), split='val', region='all', sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=import_from_path('val', config)) 
        dt_test     = SEN12MSCRTS(os.path.expanduser(config.root2), split='test', region='all', sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=import_from_path('test', config))

    # wrap to allow for subsampling, e.g. for test runs etc
    dt_train    = torch.utils.data.Subset(dt_train, range(0, min(config.max_samples_count, len(dt_train), int(len(dt_train)*config.max_samples_frac))))
    dt_val      = torch.utils.data.Subset(dt_val, range(0, min(config.max_samples_count, len(dt_val), int(len(dt_train)*config.max_samples_frac))))
    dt_test     = torch.utils.data.Subset(dt_test, range(0, min(config.max_samples_count, len(dt_test), int(len(dt_train)*config.max_samples_frac))))

    # instantiate dataloaders, note: worker_init_fn is needed to get reproducible random samples across runs if vary_samples=True
    train_sampler = DistributedSampler(dt_train, shuffle=True,
                                    seed=config.rdm_seed) if config.dist else None
    train_loader  = DataLoader(dt_train,
                            batch_size=config.batch_size,
                            sampler=train_sampler,
                            shuffle=(train_sampler is None),
                            num_workers=config.num_workers,
                            pin_memory=True,
                            worker_init_fn=seed_worker)

    val_sampler   = DistributedSampler(dt_val, shuffle=False) if config.dist else None
    val_loader    = DataLoader(dt_val, batch_size=config.batch_size,
                            sampler=val_sampler,
                            shuffle=False, num_workers=config.num_workers, pin_memory=True)

    test_sampler  = DistributedSampler(dt_test, shuffle=False) if config.dist else None
    test_loader   = DataLoader(dt_test, batch_size=config.batch_size,
                               sampler=test_sampler,
                            shuffle=False, num_workers=config.num_workers, pin_memory=True)

    print("Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test)))

    # model definition
    # (compiled model hangs up in validation step on some systems, retry in the future for pytorch > 2.0)
    # Build model and move to current GPU
    if config.dist:
        torch.cuda.set_device(config.local_rank)
    
    # 添加这行定义 device
    device = torch.device("cuda", config.local_rank if config.dist else 0)
    
    # 其他初始化代码...
    prepare_output(config)
    model = get_model(config).to(device)

    # (Optional) Wrap with DDP
    if config.dist:
        model = DDP(model,
                    device_ids=[config.local_rank],
                    output_device=config.local_rank,
                    find_unused_parameters=True)

    # Get the actual network (DDP wrapper contains .module)
    core = model.module if config.dist else model

    # Set custom attributes & print parameters
    core.len_epoch = len(train_loader)
    config.N_params = utils.get_ntrainparams(core)

    if is_master:
        print("\n\nTrainable layers:")
        for name, p in core.named_parameters():
            if p.requires_grad:
                print(f"\t{name}")

    # Random weight initialization
    print('\nInitializing weights randomly.')
    core.netG.apply(weight_init)
    
    if config.trained_checkp and len(config.trained_checkp)>0:
        # load weights from the indicated checkpoint
        print(f'Loading weights from (pre-)trained checkpoint {config.trained_checkp}')
        real_model = model.module if isinstance(model, DistributedDataParallel) else model
        
        load_model(config, real_model, train_out_layer=True, load_out_partly=...)

    if is_master:
        with open(os.path.join(config.res_dir, config.experiment_name, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))
        print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
        print(model)

    # Optimizer and Loss
    model.criterion = losses.get_loss(config)

    # track best loss, checkpoint at best validation performance
    is_better, best_loss = lambda new, prev: new <= prev, float("inf")

    # Training loop
    trainlog = {}

    # resume training at scheduler's latest epoch, != 0 if --resume_from
    begin_at = config.resume_at if config.resume_at >= 0 else model.scheduler_G.state_dict()['last_epoch']
    for epoch in range(begin_at+1, config.epochs + 1):
        if config.dist:
            train_sampler.set_epoch(epoch)
        print("\nEPOCH {}/{}".format(epoch, config.epochs))

        # put all networks in training mode again
        model.train()
        if config.dist:
            model.module.netG.train()
        else:
            model.netG.train()

        # unfreeze all layers after specified epoch
        if epoch>config.unfreeze_after and hasattr(core, 'frozen') and core.frozen:
            print('Unfreezing all network layers')
            core.frozen = False
            freeze_layers(core.netG, grad=True)

        # re-seed train generator for each epoch anew, depending on seed choice plus current epoch number
        if config.vary_samples:
            # condition dataloader samples on current epoch count
            f.manual_seed(config.rdm_seed + epoch)
            train_sampler = DistributedSampler(dt_train, shuffle=True,
                                            seed=config.rdm_seed + epoch) if config.dist else None
            train_loader = DataLoader(
                            dt_train,
                            batch_size=config.batch_size,
                            sampler=train_sampler,
                            shuffle=(train_sampler is None),
                            worker_init_fn=seed_worker, generator=f,
                            num_workers=config.num_workers,
                            pin_memory=True)

        train_metrics = iterate(
            model,
            data_loader=train_loader,
            config=config,
            writer=writer,
            mode="train",
            epoch=epoch,
            device=device,
        )
        
        # do regular validation steps at the end of each training epoch
        if epoch % config.val_every == 0 and epoch > config.val_after:
            print("Validation . . . ")

            model.eval()
            if config.dist:
                model.module.netG.eval()
            else:
                model.netG.eval()

            val_metrics, val_img_metrics = iterate(
                                            model,
                                            data_loader=val_loader,
                                            config=config,
                                            writer=writer,
                                            mode="val",
                                            epoch=epoch,
                                            device=device,
                                        )
            # use the training loss for validation
            print('Using training loss as validation loss')
            if "val_loss" in val_metrics: val_loss = val_metrics["val_loss"]
            else: val_loss = val_metrics['val_loss_ensembleAverage']

            
            print(f'Validation Loss {val_loss}')
            print(f'validation image metrics: {val_img_metrics}')
            if is_master:
                save_results(val_img_metrics, os.path.join(config.res_dir, config.experiment_name), split=f'val_epoch_{epoch}')
                print(f'\nLogged validation epoch {epoch} metrics to path {os.path.join(config.res_dir, config.experiment_name)}')   

            # checkpoint best model
            trainlog[epoch] = {**train_metrics, **val_metrics}
            if is_master:
                checkpoint(trainlog, config)
                if is_better(val_loss, best_loss):
                    best_loss = val_loss
                    real_model = model.module if isinstance(model, DistributedDataParallel) else model
                    save_model(config, epoch, real_model, "model")
        else:
            trainlog[epoch] = {**train_metrics}
            if is_master:
                checkpoint(trainlog, config)

        # always checkpoint the current epoch's model
        if is_master:
            real_model = model.module if isinstance(model, DistributedDataParallel) else model
            save_model(config, epoch, real_model, f"model_epoch_{epoch}")

        print(f'Completed current epoch of experiment {config.experiment_name}.')

    # Only master rank runs testing
    if is_master:
        print("Testing best epoch . . .")

        real_model = model.module if isinstance(model, DistributedDataParallel) else model
        load_checkpoint(config, config.res_dir, real_model, "model")
  # following training, test on hold-out data
    
        model.eval()
        if config.dist:
            model.module.netG.eval()
        else:
            model.netG.eval()

        test_metrics, test_img_metrics = iterate(
                                    model,
                                    data_loader=test_loader,
                                    config=config,
                                    writer=writer,
                                    mode="test",
                                    epoch=epoch,
                                    device=device,
                                )

        if "test_loss" in test_metrics: test_loss = test_metrics["test_loss"]
        else: test_loss = test_metrics['test_loss_ensembleAverage']
        print(f'Test Loss {test_loss}')
        print(f'\nTest image metrics: {test_img_metrics}')
        save_results(test_img_metrics, os.path.join(config.res_dir, config.experiment_name), split='test')
        print(f'\nLogged test metrics to path {os.path.join(config.res_dir, config.experiment_name)}')   

    # close tensorboard logging
    writer.close()

    print(f'Finished training experiment {config.experiment_name}.')

if __name__ == "__main__":
    main(config)
    exit()