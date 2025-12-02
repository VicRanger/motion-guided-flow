import torch
from utils.buffer_utils import flow_to_motion_vector
from utils.str_utils import dict_to_string
from utils.log import log
# from mmcv.ops.point_sample import bilinear_grid_sample, point_sample

check = True

def get_merged_motion_vector_from_last(last_mv, merged_mv, residual, method="nearest", with_batch=True, onnx=False, force_dtype_conversion=False):
    '''
    last_mv: last (i+1)^{th} mv
    merged_mv: last i^{th} mv using to warp last i^{th} frame to current frame
    '''
    assert method == 'nearest' or method == 'bilinear'
    if not with_batch and check:
        if len(last_mv.shape) == 3:
            last_mv = last_mv.unsqueeze(0)
        if merged_mv is not None and len(merged_mv.shape) == 3:
            merged_mv = merged_mv.unsqueeze(0)
        if residual is not None and len(residual.shape) == 3:
            residual = residual.unsqueeze(0)
    # log.debug(dict_to_string(warp(last_mv, merged_mv, mode="nearest", padding_mode="zeros"), "warped last_mv" ,mmm=True))
    # log.debug(dict_to_string(merged_mv, "merged_mv input" ,mmm=True))
    if merged_mv is None:
        assert residual is not None
        return last_mv + residual
    if residual is None:
        residual = merged_mv
    ret = warp(last_mv, merged_mv, mode=method,  padding_mode="zeros", force_dtype_conversion=force_dtype_conversion) + residual
    # log.debug(dict_to_string(ret, "merged mv ouptut", mmm=True))
    return ret


flow_base_storage = {}
def get_normalized_coords(flow=None, device=None, dtype=None, B=None, H=None, W=None, align_corners=True):
    if flow is not None:
        device = flow.device
        dtype = flow.dtype
        B = flow.shape[0]
        H = flow.shape[2]
        W = flow.shape[3]
    else:
        assert (device is not None and dtype is not None and B is not None and H is not None and W is not None)
    
    k = (str(device), str(dtype), f"{B}x{H}x{W}")
    # k = (str(flow.device), str(flow.size()), str(flow.type))
    if align_corners:
        offset_W, offset_H = 0, 0
    else: 
        offset_W, offset_H = 1/(W*2), 1/(H*2)
    if k not in flow_base_storage.keys():
        hori = torch.linspace(-1.0+offset_W, 1.0-offset_W, W, dtype=dtype).view(
            1, 1, 1, W).expand(B, -1, H, -1)
        verti = torch.linspace(-1.0+offset_H, 1.0+offset_H, H, dtype=dtype).view(
            1, 1, H, 1).expand(B, -1, -1, W)
        g = torch.cat([hori, verti], 1)
        g = g.to(device)
        flow_base_storage[k] = g
    g = flow_base_storage[k]
    return g

def batch_warp(imgs, flows, flow_type="mv", mode="bilinear", padding_mode="border", align_corners=True, force_dtype_conversion=False) -> tuple:
    assert len(imgs) == len(flows)
    warped_imgs = warp(torch.cat(imgs, dim=0), torch.cat(flows, dim=0), flow_type=flow_type, mode=mode, padding_mode=padding_mode, align_corners=align_corners, force_dtype_conversion=force_dtype_conversion)
    # log.debug(dict_to_string([warped_imgs, len(imgs), torch.chunk(warped_imgs, len(imgs), dim=0)]))
    return torch.chunk(warped_imgs, len(imgs), dim=0)

def warp(img, flow, flow_type="mv", mode="bilinear", padding_mode="border", align_corners=True, force_dtype_conversion=False, onnx=False):
    '''
    flow_type (str): input flow type
        'mv' | 'flow'. Default: 'mv'
    mode (str): sample mode for warp
        'nearest' | 'bilinear'. Default: 'nearest'
    padding_mode (str): padding mode for outside grid values
        'zeros' | 'border' | 'reflection'. Default: 'zeros'
    '''
    if check:
        if img.device != flow.device:
            print("warp function deal with two different device.")
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(flow.shape) == 3:
            flow = flow.unsqueeze(0)
    # print(img.shape, flow.shape)
    assert img.shape[2] == flow.shape[2] and img.shape[3] == flow.shape[3], f"img.shape={img.shape}, flow.shape={flow.shape}"
    g = get_normalized_coords(flow, align_corners=align_corners)
    assert g.dtype == flow.dtype, f"g.dtype={g.dtype}, flow.dtype={flow.dtype}"
    # g = g.type_as(flow)
    # log.debug("g.dtype: {}, flow.dtype:{}".format(g.dtype, flow.dtype))
    if flow_type == "flow":
        flow = flow_to_motion_vector(flow)
    flow = g - flow
    flow = flow.permute(0, 2, 3, 1)
    # log.debug(dict_to_string(img, "img", mmm=True))
    # log.debug(dict_to_string(flow, "flow", mmm=True))
    # onnx = False
    # if onnx:
    #     assert mode in ['nearest', 'bilinear']
    #     if mode == 'nearest':
    #         return point_sample( img, flow * 0.5 + 0.5, align_corners=True)
    #     else:
    #         return bilinear_grid_sample(img, flow, align_corners=True)
    # else:
    ''' dont change dtype when computation graph enabled '''
    if force_dtype_conversion and img.dtype != flow.dtype:
        img = img.type(flow.dtype)
    # assert img.dtype == flow.dtype, f"img.dtype={img.dtype}, flow.dtype={flow.dtype}"
    # log.debug(dict_to_string([img, flow]))
    return torch.nn.functional.grid_sample(input=img, grid=flow,
                                        mode=mode, padding_mode=padding_mode,
                                    align_corners=align_corners)
