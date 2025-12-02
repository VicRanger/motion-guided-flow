import copy
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from trainers.fe_trainer_base import FETrainerBase, get_his_recurrent_list
from dataloaders.raw_data_importer import get_augmented_buffer
from utils.buffer_utils import data_as_type
from utils.loss_utils import lpips, psnr, ssim
from utils.utils import del_data
import torch
from trainers.trainer_base import TrainerBase
from models.loss.loss import LossFunction
from utils.buffer_utils import aces_tonemapper, align_channel_buffer, buffer_data_to_vis, inv_log_tonemapper, to_numpy
from utils.str_utils import dict_to_string, dict_to_string_join
from utils.log import get_local_rank, log
from utils.warp import get_merged_motion_vector_from_last, warp

import json
import numpy as np
import torch
# from code.src.utils.utils import del_dict_item
from utils.str_utils import dict_to_string
from utils.log import log
import re
import os.path as osp

class MFRRNetTrainer(FETrainerBase):
    def gather_tensorboard_image(self, mode='train'):
        diff_scale = 4
        self.add_render_buffer("pred", buffer_type='scene_color')
        self.add_render_buffer("gt", buffer_type='scene_color')
        pred = aces_tonemapper(self.get_buffer("pred", allow_skip=False))
        gt = aces_tonemapper(self.get_buffer("gt", allow_skip=False))
        diff = diff_scale * ((pred - gt)**2)
        self.add_render_buffer(f"diff ({diff_scale}x)", buffer=diff)
        if 'pred_st_color' in self.cur_output.keys():
            pred_no_st = aces_tonemapper(self.get_buffer("pred_scene_color_no_st", allow_skip=False))
            gt_no_st = aces_tonemapper(self.get_buffer("scene_color_no_st", allow_skip=False))
            diff = diff_scale * ((pred_no_st - gt_no_st)**2)
            self.add_render_buffer("pred_scene_color_no_st", buffer_type='scene_color')
            self.add_render_buffer("scene_color_no_st", buffer_type='scene_color')
            self.add_render_buffer(f"diff_no_st ({diff_scale}x)", buffer=diff)
            pred_st = aces_tonemapper(self.get_buffer("pred_st_color", allow_skip=False))
            gt_st = aces_tonemapper(self.get_buffer("st_color", allow_skip=False))
            diff = diff_scale * ((pred_st - gt_st)**2)
            self.add_render_buffer("pred_st_color", buffer_type='scene_color')
            self.add_render_buffer("st_color", buffer_type='scene_color')
            self.add_render_buffer(f"diff_st ({diff_scale}x)", buffer=diff)
            if mode == 'test':
                self.prefix_texts.insert(0, f'lpips_st: {float(lpips(pred_st, gt_st)):.4g}')
            self.prefix_texts.insert(0, f'ssim_st: {ssim(pred_st, gt_st):.4g}')
            self.prefix_texts.insert(0, f'psnr_st: {psnr(pred_st,gt_st):.4g}')
            if mode == 'test':
                self.prefix_texts.insert(0, f'lpips_no_st: {float(lpips(pred_no_st, gt_no_st)):.4g}')
            self.prefix_texts.insert(0, f'ssim_no_st: {ssim(pred_no_st, gt_no_st):.4g}')
            self.prefix_texts.insert(0, f'psnr_no_st: {psnr(pred_no_st, gt_no_st):.4g}')
        if mode == 'test':
            self.prefix_texts.insert(0, f'lpips: {float(lpips(pred, gt)):.4g}')
        self.prefix_texts.insert(0, f'ssim: {ssim(pred, gt):.4g}')
        self.prefix_texts.insert(0, f'psnr: {psnr(pred, gt):.4g}')


    def __init__(self, config, model, resume=False):
        super().__init__(config, model, resume)
        self.output_cache = None
        self.last_output = []
        self.last_scene_name = ""
        self.last_index = -1
        
    def get_model_loss(self):
        log.debug(dict_to_string(f'get model loss using "-psnr"'))
        loss = self.get_avg_info("psnr")
        assert loss is not None
        loss *= -1.0
        return loss
    
    def gather_tensorboard_image_debug(self, mode='train') -> None:
        with torch.no_grad():
            num_he = int(self.model.get_net().num_history_encoder)  # type: ignore
            num_dec = int(self.model.get_net().num_shade_decoder_layer)  # type: ignore

            net = self.model.get_net()
            self.add_render_buffer("pred", buffer_type="scene_color", debug=True)
            self.add_render_buffer("scene_color", buffer_type="scene_color", debug=True)

            if net.enable_demodulate:
                albedo = self.get_buffer('dmdl_color', allow_skip=False)
                self.add_render_buffer(f"dmdl_color({self.config['dataset']['demodulation_mode']})", albedo, debug=True)
            else:
                albedo = None

            if self.model.get_net().method in ["residual", "shade"]:
                self.add_render_buffer("pred_scene_light_no_st", debug=True)
                self.add_render_buffer("scene_light_no_st", debug=True)
                self.add_render_buffer("disc_mask", buffer_type="depth", debug=True)
                self.add_render_buffer("residual_mask", buffer_type="depth", debug=True)
                residual_item = self.get_buffer("residual_item", allow_skip=False)
                self.add_render_buffer("abs(residual)", buffer=torch.abs(-(residual_item) +  # type: ignore
                                                                         self.get_buffer("pred_scene_light_no_st", allow_skip=False)),
                                       buffer_type="depth", debug=True)
                self.add_render_buffer("pred_warped_scene_color_no_st", buffer=residual_item, buffer_type="scene_color", debug=True)
                self.add_diff_buffer("gt_comp", "gt", debug=True)
                self.add_render_buffer(f'pred_layer_{0}_tmv_{0}', buffer_type="motion_vector_8", debug=True)

            if self.model.get_net().enable_st:
                self.add_render_buffer("pred_st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_st_alpha", debug=True)
                self.add_render_buffer("st_alpha", debug=True)
                self.add_render_buffer("pred_sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("skybox_mask", debug=True)
                self.add_render_buffer("pred_comp_color_before_sky_st", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_comp_color_sky", buffer_type="scene_color", debug=True)

            def get_pyramid_buffer(layer_id, he_id, in_name):
                if f'pred_layer_{layer_id}_{in_name}_{he_id}' not in self.cur_output.keys():
                    return None
                if i == num_dec:
                    mv = self.cur_output[f'pred_layer_{layer_id}_{in_name}_{he_id}'][0]
                else:
                    ratio = 2 ** (layer_id)
                    mv = F.interpolate(self.cur_output[f'pred_layer_{layer_id}_{in_name}_{he_id}'][:1], scale_factor=ratio)[0]
                return mv

            for he_id in range(num_he):
                if self.model.get_net().enable_lmv_res:
                    for i in range(num_dec):
                        if he_id == 0:
                            if self.model.get_net().enable_lmv_res:
                                self.add_render_buffer(f'l{i}_lmv_res_{he_id}', buffer_type="motion_vector_64",
                                                       buffer=get_pyramid_buffer(i, he_id, "lmv_res"), debug=True)
                            if self.model.get_net().enable_st_lmv_res:
                                self.add_render_buffer(f'l{i}_st_lmv_res_{he_id}', buffer_type="motion_vector_64",
                                                       buffer=get_pyramid_buffer(i, he_id, "st_lmv_res"), debug=True)

                for i in range(num_dec):
                    if self.model.get_net().enable_feature_warp:
                        self.add_render_buffer(f'l{i}_tmv_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "tmv"), debug=True)
                    if self.model.get_net().enable_st_feature_warp:
                        self.add_render_buffer(f'l{i}_st_tmv_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "st_tmv"), debug=True)

    def __recurrent_gbuffer_layer_d2e(self, he_id):
        num_dec = int(self.net.num_shade_decoder_layer) # type: ignore
        for layer_id in range(1, num_dec):
            self.cur_data[f"history_{he_id}_ge_sc_layers_{layer_id}"] = self.last_output[-(
                he_id) - 1][f'ge_sc_layers_{layer_id}'].detach()

    def __recurrent_history_layer_d2e(self, he_id, pf=""):
        # ratio = 2 ** self.model.get_net().num_shade_decoder_layer # type: ignore
        # tmp_shape = self.cur_data['scene_color'].shape # type: ignore
        num_dec = int(self.net.num_shade_decoder_layer) # type: ignore
        for layer_id in range(1, num_dec):
            self.cur_data[f'history_{he_id}_{pf}d2e_sc_layers_{layer_id}'] = self.last_output[-(
                he_id) - 1][f'{pf}d2e_sc_layers_{layer_id}'].detach()
        # else:
        #     self.cur_data[f'history_{he_id}_{pf}d2e_sc_layers'] = [item.detach() # type: ignore
        #                                                     for item in self.last_output[-(he_id) - 1][f'd2e_sc_layers'][::-1]]

    def __recurrent_layer_d2e(self, he_id):
        self.cur_data['recurrent_d2e_he_id'] = he_id
        self.__recurrent_gbuffer_layer_d2e(he_id)
        self.__recurrent_history_layer_d2e(he_id)

    def __recurrent_one_batch_data(self, he_id):
        history_datas = self.cur_data['history_data_list']
        pred_buffres = ['scene_color_no_st']
        if self.net.enable_st:
            pred_buffres += ['st_color', 'st_alpha', 'sky_color']
        # log.debug(pred_buffres)
        for buffer_name in pred_buffres:
            history_datas[he_id][f'{buffer_name}'] = self.last_output[-1 - he_id][f'pred_{buffer_name}'].detach()  # type: ignore
        # self.cur_data[f'history_scene_color_no_st_{he_id}'] = self.last_output[-1-he_id]['pred_scene_color_no_st'].detach() # type: ignore
        self.cur_data[f'recurrent_pred_{he_id}'] = True
        he_pf = self.net.he_pfs[he_id]  # type: ignore
        assert not f"{he_pf}sc_layers" in self.cur_data.keys()
        
    def set_recurrent_data(self, mode='train'):
        ''' used in ShadeTrainerBase.load_data() '''
        ''' run before self.model.get_augment_data() '''
        num_he = self.config['model']['history_encoders']['num']
        full_rendered = True
        recurrent_pred = (self.config['trainer'][f'recurrent_{mode}'])
        ''' create feature_0 and encoding_0 '''
        his_recurrent_list = get_his_recurrent_list(cur_data_index=self.cur_data_index,
                                                    num_he=num_he,
                                                    block_size=self.get_block_size(mode))
        for he_id in range(num_he):
            he_pf = self.net.he_pfs[he_id]  # type: ignore
            self.cur_data[he_pf + "prob"] = 0  # type: ignore
            if self.cur_data_index <= he_id:
                continue
            if his_recurrent_list[he_id]:
                if recurrent_pred:
                    start_recurrent_epoch = int(self.end_epoch * self.config['trainer']['recurrent_train_start'])
                    if self.epoch_index == start_recurrent_epoch and self.batch_index == 0:
                        self.min_loss = 1e9
                    if not (mode == "train" and self.epoch_index < start_recurrent_epoch):
                        self.__recurrent_one_batch_data(he_id)
                        self.cur_data[he_pf + "prob"] = 1  # type: ignore
                        full_rendered = False
            else:
                ...
                # log.debug(f"@{self.cur_data_index}.{he_id}, no recurrent")
        self.cur_data['rendered_prob'] = 1 if full_rendered else 0  # type: ignore
        self.cur_data['trainer_mode'] = mode
        
    def set_recurrent_feature(self, mode='train'):
        num_he = self.config['model']['history_encoders']['num']
        his_recurrent_list = get_his_recurrent_list(cur_data_index=self.cur_data_index,
                                                    num_he=num_he,
                                                    block_size=self.get_block_size(mode))

        if self.net.enable_recurrent_d2e:
            self.cur_data['recurrent_d2e_he_id'] = -1
        for he_id in range(num_he):
            if self.cur_data_index <= he_id:
                continue
            ''' streaming from decoder must occur when the last frame is predicted '''
            if his_recurrent_list[he_id]:
                # log.debug(f"@{self.cur_data_index}.{he_id}, recurrent")
                ''' recurrent_d2e '''
                if self.net.enable_recurrent_d2e and self.cur_data['recurrent_d2e_he_id'] == -1:
                    self.__recurrent_layer_d2e(he_id)
                # and self.model.get_net().enable_recurrent \
            else:
                ...
                
    def update(self, data, epoch_index=None, batch_index=None, mode="train"):
        ''' check if its same block with last_data '''
        # log.debug(dict_to_string([data['metadata'] for data in datas]))
        for i, item in enumerate(data):
            # if get_local_rank() == 0:
            #     if len(self.last_output) > 0:
            #         log.debug("{} {}".format(self.last_output[-1]['metadata'], data['metadata']))
            is_same_block = self.last_index > 0 and self.last_scene_name == item['metadata']['scene_name'][0] \
                and int(self.last_index) == int(item['metadata']['index'][0]) - 1
            self.last_scene_name = item['metadata']['scene_name'][0]
            self.last_index = item['metadata']['index'][0]
            # log.debug(dict_to_string(self.last_output, 'self.last_output', max_depth=0))
            if is_same_block:
                ''' continue the current block '''
                self.cur_data_index += 1
                # if get_local_rank() == 0:
                #     log.debug(f"continue with {data['metadata']['index'][0]}, index: {self.cur_data_index}.")
            else:
                ''' start a new block '''
                self.cur_data_index = 0
                del self.last_output
                self.last_output = []
                # if get_local_rank() == 0:
                #     log.debug(f"new block started. index: {self.cur_data_index}, with {data['metadata']['index'][0]}")
            # log.debug(f'self.cur_data_index: {self.cur_data_index}')
            self.load_data(item, mode)
            self.update_one_batch(epoch_index=epoch_index, batch_index=batch_index, mode=mode)
            # log.debug(dict_to_string(data))
            

    def cache_one_batch_output(self, mode, epoch_index=None, batch_index=None):
        recurrent_pred = (self.config['trainer'][f'recurrent_{mode}'])
        num_he = int(self.net.num_history_encoder)  # type: ignore
        num_dec = int(self.net.num_shade_decoder_layer) # type: ignore
        if len(self.last_output) > num_he:
            del self.last_output[0]
        cache_output = {}
        if recurrent_pred:
            cache_output.update({'pred_'+k: self.cur_output['pred_'+k] for k in self.config['model']['pred_buffers']})
            cache_output['pred_scene_color_no_st'] = self.cur_output['pred_scene_color_no_st']
        # for he_id in range(num_he-1):
        #     cache_output[f'{self.net.he_pfs[he_id]}sc_layers'] = [item for item in self.cur_output[f'{self.net.he_pfs[he_id]}sc_layers_cached']]
        #     cache_output[f'{self.net.he_pfs[he_id]}output'] = self.cur_output[f'{self.net.he_pfs[he_id]}output_cached']
        for layer_id in range(1, num_dec):
            if self.net.enable_recurrent_d2e:
                cache_output[f'd2e_sc_layers_{layer_id}'] = self.cur_output[f'd2e_sc_layers_{layer_id}']
                cache_output[f'ge_sc_layers_{layer_id}'] = self.cur_output[f'ge_sc_layers_{layer_id}']

        cache_output['metadata'] = copy.deepcopy(self.cur_data['metadata'])  # type: ignore
        self.last_output.append(cache_output)