
from pytorch_msssim.ssim import ssim as pytorch_msssim_ssim
from pytorch_msssim.ssim import ms_ssim as pytorch_msssim_msssim
from utils.buffer_utils import aces_tonemapper
from lpips import LPIPS
import torch
from utils.log import log
from utils.str_utils import dict_to_string

lpips_kernel_caches = {}


def lpips(pred, gt, normalize=True):
    assert pred.device == gt.device
    assert pred.dtype == gt.dtype, dict_to_string([str(pred.dtype), str(gt.dtype)])
    cache_key = f"device_{pred.device}_dtype_{pred.dtype}"
    global lpips_kernel
    if not (lpips_kernel:=lpips_kernel_caches.get(cache_key)):
        lpips_kernel = LPIPS(net='vgg', verbose=False).to(pred.device).type(pred.dtype)
        lpips_kernel_caches[cache_key] = lpips_kernel
    assert isinstance(lpips_kernel, LPIPS), dict_to_string([lpips_kernel, str(type(lpips_kernel))])
    return lpips_kernel(pred, gt, normalize=normalize)

def msssim(pred, gt, size_average=True):
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    return pytorch_msssim_msssim(pred, gt, data_range=1, size_average=size_average)

def ssim(pred, gt, size_average=True):
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    return pytorch_msssim_ssim(pred, gt, data_range=1, size_average=size_average)

def dssim(pred, gt, size_average=True):
    return 1-ssim(pred, gt, size_average=size_average)

def dmsssim(pred, gt, size_average=True):
    return 1-msssim(pred, gt, size_average=size_average)


def psnr(pred, gt, **kwargs):
    mse = (torch.clamp(pred, 0, 1) - torch.clamp(gt, 0, 1)).square().mean()
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

def psnr_mask(pred, gt, mask, **kwargs):
    mse = (torch.clamp(pred, 0, 1)*mask - torch.clamp(gt, 0, 1)*mask).square().mean() / mask.mean()
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

def normalized_psnr(pred, gt, **kwargs):
    diff_map = torch.square(torch.clamp(pred, 0, 1) - torch.clamp(gt, 0, 1))
    se_map = diff_map.sum(dim=1, keepdim=True)
    se = se_map.sum()
    se_cnt = torch.where(se_map > 0, torch.ones_like(se_map), torch.zeros_like(se_map)).sum() + 1
    # log.debug(dict_to_string([se, se_cnt]))
    psnr = 10 * torch.log10(1.0 / (se/se_cnt))
    return psnr

def psnr_hdr(pred, gt, **kwargs):
    return psnr(aces_tonemapper(pred), aces_tonemapper(gt))