import copy
import time
from sympy import use
import torch
from torch import optim
from torch.amp.autocast_mode import autocast as autocast
from utils.buffer_utils import aces_tonemapper, inv_log_tonemapper, inv_rein_tonemapper
from dataloaders.raw_data_importer import tensor_as_type_str
from utils.dataset_utils import data_to_device, data_to_device_dict
from utils.str_utils import dict_to_string
from utils.log import get_local_rank, log
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from utils.model_utils import get_1d_dim, get_2d_dim, get_model_parm_nums, model_to_half
from utils.timer import Timer
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

def module_load_by_path(module, path):
    path = Path(path) # Ensure path is a Path object for consistent handling
    if not path.exists():
        raise FileNotFoundError(f'No file found at "{path}"')
    loaded_successfully = False
    error_messages = []
    net_data = None
    
    try:
        # print(f"Attempting to load '{path}' with weights_only=True...")
        net_data = torch.load(path, weights_only=True)
        if isinstance(net_data, dict) and all(isinstance(k, str) for k in net_data.keys()):
            module.load_state_dict(net_data)
            loaded_successfully = True
        else:
            error_messages.append(f"Data type after loading with weights_only=True is unexpected ({type(net_data)}), expected a dictionary.")

    except Exception as e:
        error_messages.append(f"Loading with weights_only=True failed: {type(e).__name__}: {e}")
    
    if loaded_successfully:
        return
    
    if not loaded_successfully:
        raise RuntimeError(
            f"Failed to load model weights from '{path}'. "
            f"Attempts with weights_only=True and weights_only=False both failed. "
            f"Detailed error messages: {'; '.join(error_messages)}"
        )

def get_input_by_type_str(data, t="fp32") -> dict:
    model_input = {}
    for k in data.keys():
        if isinstance(data[k], torch.Tensor):
            model_input[k] = tensor_as_type_str(
                data[k], t)
        elif isinstance(data[k], list):
            tmp_arr = data[k]
            if isinstance(tmp_arr[0], torch.Tensor):
                model_input[k] = tmp_arr
                for i in range(len(model_input[k])):
                    model_input[k][i] = tensor_as_type_str(
                        model_input[k][i], t)
        else:
            model_input[k] = data[k]
    return model_input


class ModelBase:
    def __init__(self, config: dict):
        self.trainer_config = config
        self.config = config['model']
        self.config['dataset'] = config['dataset']
        self.net = None
        self.ddp = False
        self.infer_time = -1
        self.use_cuda = self.trainer_config['num_gpu'] > 0
        self.dummy_output = None
        self.model_name = self.config['model_name']
        self.inference_precision = self.config['inference_precision']
        self.inference_precision_mode = torch.float16 if self.config['inference_precision'] == "fp16" else torch.float
        self.instance_name = "{}({})".format(
            self.config['class'], self.model_name)
        log.info("[{}] model creating...".format(self.instance_name))
        log.info(f"inference_precision: {self.inference_precision}")
        if self.trainer_config['use_cuda']:
            self.use_cuda = True
        self.create_model()
        if self.inference_precision == 'fp16':
            self.net = model_to_half(self.net)
        self.to_device()
        self.dummy_input_size_h = self.config.get('dummy_input_size_h', 720)
        self.dummy_input_size_w = self.config.get('dummy_input_size_w', 1280)

        dummy_data = self.get_dummy_input(bs=1)
        if self.inference_precision == 'fp16':
            dummy_data = get_input_by_type_str(dummy_data, "fp16")
        if self.use_cuda:
            dummy_data = data_to_device_dict(dummy_data, device=self.trainer_config['device'])
        self.dummy_input = dummy_data
        self.dummy_net_input = self.calc_preprocess_input(self.dummy_input)
        log.debug(dict_to_string(self.dummy_net_input))
        self.set_eval()
        self.get_net().enable_timing = True  # type: ignore
        self.dummy_output = self.update(copy.deepcopy(self.dummy_net_input))
        self.get_net().enable_timing = False  # type: ignore
        log.debug(dict_to_string(self.dummy_output))
        if self.trainer_config.get("initial_inference", True) and get_local_rank() == 0:
            self.run_dummy_inference()
        log.info("[{}] model created.".format(self.instance_name))

    def create_model(self):
        pass

    def get_dummy_input(self, input_2d_str, input_1d_str, bs=1) -> dict:
        H, W = self.dummy_input_size_h, self.dummy_input_size_w
        ret = {}
        for item in input_2d_str:
            if item.startswith('d2_'):
                tmp_tensor = torch.zeros(1, get_2d_dim(item), H//2, W//2)
            else:
                tmp_tensor = torch.zeros(1, get_2d_dim(item), H, W)
            if self.use_cuda:
                tmp_tensor = tmp_tensor.cuda()
            ret[item] = tmp_tensor
        for item in input_1d_str:
            tmp_tensor = torch.zeros(1, get_1d_dim(item))
            if self.use_cuda:
                tmp_tensor = tmp_tensor.cuda()
            ret[item] = tmp_tensor
        ret['cur_data_index'] = 0
        ret['metadata'] = {
            'scene_name': ['scene_name'],
            'index': [0],
        }
        return ret

    def get_augment_data(self, data: dict) -> dict:
        return data

    def run_dummy_inference(self):
        # log.debug(self.net)

        input_data = self.dummy_net_input
        self.dummy_output = self.inference_for_timing(input_data)
        num_run = 1000
        num_warm_run = 300
        log.info("[{}] start dummy_inference...".format(self.instance_name))
        log.info("[{}] warm_run:{}, run:{}".format(self.instance_name, num_warm_run, num_run))
        if self.trainer_config.get('debug', False):
            num_run = num_warm_run = 1
        with torch.no_grad():
            for _ in tqdm(range(num_warm_run)):
                # input_data = copy.deepcopy(self.dummy_net_input)
                self.inference_for_timing(input_data)
                torch.cuda.synchronize()
            timer = Timer()
            for _ in tqdm(range(num_run)):
                # input_data = copy.deepcopy(self.dummy_net_input)
                torch.cuda.synchronize()
                timer.start()
                self.inference_for_timing(input_data)
                torch.cuda.synchronize()
                timer.stop()
        self.infer_time = timer.get_avg_time()
        log.info("[{}] dummy_inference completed.".format(self.instance_name))
        log.info("estimated infer_time ({} runs): {}".format(num_run, (self.infer_time)))
        # exit(0)

    # only for time testing
    def inference_for_timing(self, data):
        self.set_eval()
        with torch.no_grad():
            output = self.get_net()(data)
        return output

    def inference(self, data):
        self.set_eval()
        with torch.no_grad():
            output = self.update(data, training=False)
        return output

    def update(self, data):
        net_input = self.calc_preprocess_input(data)
        output = self.forward(net_input)
        return output

    def calc_preprocess_input(self, data) -> dict:
        return data

    def forward(self, net_input):
        assert self.net is not None
        output = self.net(net_input)
        return output

    def calc_net_output(self, data):
        return data

    def calc_loss(self, data, ret):
        pass

    def load_by_dict(self, state_dict):
        self.get_net().load_state_dict(state_dict)

    def load_by_path(self, path):
        module_load_by_path(self.get_net(), path)

    def save(self, path, state_dict_only=True):
        torch.save(self.get_net().state_dict(), path)
            
    def get_net(self) -> nn.Module:
        assert self.net is not None
        if self.ddp:
            assert self.net.module is not None
            return self.net.module
        else:
            return self.net

        # log.debug("[{}] model saved.".format(self.model_name))

    def to_device(self):
        assert self.net is not None
        if self.use_cuda:
            self.net.cuda(self.trainer_config['device'])

    def to_ddp(self, device_ids=[], output_device=0, find_unused_parameters=True):
        if self.ddp:
            raise Exception("Already a ddp model, please check code.")
        if len(device_ids) == 0:
            device_ids.append(output_device)
        # self.net = DDP(self.net, device_ids=device_ids, output_device=output_device, find_unused_parameters=True)
        self.net = DDP(self.net, device_ids=device_ids, output_device=output_device,
                       find_unused_parameters=self.trainer_config['trainer']['ddp__find_unused_parameters'])
        self.ddp = True

    def get_infer_time(self):
        return self.infer_time

    def get_net_parameter_num(self):
        return get_model_parm_nums(self.get_net())

    def set_train(self):
        net = self.get_net()
        net.train()

    def set_eval(self):
        net = self.get_net()
        net.eval()


class ModelBaseEXT(ModelBase):

    def __init__(self, config: dict):
        ModelBase.__init__(self, config)

    def set_train(self):
        net = self.get_net()
        net.set_train()

    def set_eval(self):
        net = self.get_net()
        net.set_eval()


inv_tonemap = inv_rein_tonemapper


class BlendModelBase(ModelBase):

    def __init__(self, config: dict):
        self.enable_blend = config['model']['config'].get('enable_blend_mode', False)
        log.debug(f'BlendModelBase.enable_blend_mode: {self.enable_blend}')
        ModelBase.__init__(self, config)

    def update(self, data):
        if not self.enable_blend:
            net_input = self.calc_preprocess_input(data)
            output = self.forward(net_input)
            # output['gt'] = inv_tonemap(output['gt'])
            # output['scene_color'] = inv_gamma_log(output['scene_color'])
            # output['pred'] = inv_tonemap(output['pred'])
        else:
            data = self.calc_preprocess_input(data)
            base_input = {
                'metadata': data['metadata'],
                'time': data['time'],
            }
            net_input0 = {
                'img0': data[self.config['st_alpha_0_alias']].repeat(1, 3, 1, 1),
                'img1': data[self.config['st_alpha_1_alias']].repeat(1, 3, 1, 1),
                'gt': data['st_alpha'].repeat(1, 3, 1, 1),
                **base_input,
            }
            res0 = self.forward(net_input0)
            pred_st_alpha = res0['pred'][:, :1].detach()
            torch.cuda.empty_cache()

            net_input1 = {
                'img0': data[self.config['st_color_0_alias']],
                'img1': data[self.config['st_color_1_alias']],
                'gt': data['st_color'],
                **base_input,
            }
            res1 = self.forward(net_input1)
            pred_st_color = res1['pred'].detach()
            mv_st = res1['motion_vector'].detach()
            torch.cuda.empty_cache()

            net_input2 = {
                'img0': data[self.config['scene_color_no_st_0_alias']],
                'img1': data[self.config['scene_color_no_st_1_alias']],
                'gt': data['scene_color_no_st'],
                **base_input,
            }
            res2 = self.forward(net_input2)
            pred_scene_color_no_st = res2['pred'].detach()
            mv = res2['motion_vector'].detach()
            torch.cuda.empty_cache()

            # log.debug(dict_to_string(data, mmm=True))
            output = {}
            # output['pred_scene_color_no_st'] = inv_tonemap(pred_scene_color_no_st)
            output['pred_scene_color_no_st'] = pred_scene_color_no_st
            # output['pred_st_alpha'] = pred_st_alpha
            # output['pred_st_color'] = inv_tonemap(pred_st_color)
            output['pred_st_color'] = pred_st_color
            # output['pred'] = inv_tonemap(pred_scene_color_no_st) * pred_st_alpha + inv_tonemap(pred_st_color)
            output['pred'] = pred_scene_color_no_st * pred_st_alpha + pred_st_color
            # output['scene_color_st_blend'] = inv_gamma_log(data['scene_color_no_st']) * data['st_alpha'] + aces_tonemapper(pred_st_color)
            # output['pred'] = pred_scene_color_no_st * pred_st_alpha + pred_st_color
            # output['gt'] = inv_tonemap(data['scene_color'])
            output['gt'] = data['scene_color']
            output['scene_color'] = data['scene_color']
            # output['scene_color'] = inv_tonemap(data['scene_color'])
            # output['st_color'] = inv_tonemap(data['st_color'])
            output['st_color'] = data['st_color']
            output['motion_vector'] = mv
            output['motion_vector_st'] = mv_st
            # output['scene_color_no_st'] = inv_tonemap(data['scene_color_no_st'])
            output['scene_color_no_st'] = data['scene_color_no_st']
            output['history_scene_color_no_st_0'] = data['history_scene_color_no_st_0']
        # log.debug(dict_to_string(output, mmm=True))
        return output
