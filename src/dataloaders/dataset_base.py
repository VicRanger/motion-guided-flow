from __future__ import annotations
import time
import multiprocessing as mp
import numpy as np
# from utils.dataset_utils import motion_vector_to_flow
from torch.utils.tensorboard.summary import scalar
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.dataset_utils import DatasetGlobalConfig
from utils.log_tonemap_utils import tonemap_func
from config.config_utils import convert_to_dict
from utils.str_utils import dict_to_string
from utils.buffer_utils import fix_dmdl_color_zero_value, log_tonemapper, ldr_log_tonemapper, inv_log_tonemapper, write_buffer
from utils.log import log
import random
import torch
import glob


class MetaData:
    def __init__(self, scene_name, index):
        self.scene_name = scene_name
        self.index = index
        
    def get_offset(self, offset):
        if self.index + offset < 0:
            raise RuntimeError(
                "no offset frame for no.({}+{}={}).".format(self.index, offset, self.index + offset))
        return MetaData(self.scene_name, self.index + offset)

    def __repr__(self):
        return 'MetaData("{}",{})'.format(self.scene_name, self.index)

    def __str__(self):
        return self.__repr__()
    
    def to_dict(self):
        return {'scene_name': self.scene_name, 'index': self.index}

class MetaDataWithPath(MetaData):
    def __init__(self, scene_name: str, index: int, path_alias: str):
        super().__init__(scene_name, index)
        self.path_alias = path_alias
        
    def get_offset(self, offset):
        metadata = super().get_offset(offset)
        return MetaDataWithPath(metadata.scene_name, metadata.index, self.path_alias)
    
    def __repr__(self):
        return 'MetaDataWithPath("{}", {}, "{}")'.format(self.scene_name, self.index, self.path_alias)


def range_task_by_part_name(args):
    task, metadatas, part_name = args
    log.debug(f"start part: {part_name}")
    task(metadatas, part_name)

def dispatch_task_by_part_name(task, metadatas: list[MetaData], part_names, num_thread=0):
    ''' single thread '''
    if num_thread <= 0:
        for part_name in part_names:
            range_task_by_part_name((task, metadatas, part_name,))
        return
    ''' multi thread '''
    n_core = num_thread
    pool = mp.Pool(processes=n_core)
    try:
        pool.imap_unordered(range_task_by_part_name, [(task, metadatas, part_name) for part_name in part_names], chunksize=max(1,len(part_names)//n_core))
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
    except Exception as e:
        log.debug(e)
        pool.terminate()
    finally:
        log.debug(f"joined threads: len={len(part_names)}")
        pool.join()
        
def range_task_by_metadata(task, metadatas, start_idx, end_idx):
    time.sleep(end_idx * 0.001)
    log.debug(f"start range_task[{start_idx}:{end_idx}]")
    for i in tqdm(range(start_idx, end_idx)):
        task(metadatas[i])

def dispatch_task_by_metadata(task, metadatas: list[MetaData], num_thread=0):
    ''' single thread '''
    if num_thread <= 0:
        range_task_by_metadata(task, metadatas, 0, len(metadatas))
        return
    ''' multi thread '''
    n_core = num_thread
    num = len(metadatas)
    pool = mp.Pool(processes=n_core)
    thread_part = max(num // n_core + 1, 1)
    try:
        log.debug("scene:{} n_core:{} thread_part:{}".format(
            metadatas[0].scene_name, n_core, thread_part))
        _ = [pool.apply_async(range_task_by_metadata, (task, metadatas, i * thread_part,
                                           min((i + 1) * thread_part, num), ),
                              callback=None)
             for i in range(n_core)]
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
    except Exception as e:
        log.debug(e)
        pool.terminate()
    finally:
        log.debug(f"joined threads: len={len(metadatas)}")
        pool.join()


def create_metadata_by_glob(path, scene, part_name):
    file_name_list = glob.glob(
        "{}/{}/{}/*.npz".format(path, scene, part_name))
    num = len(file_name_list)
    metadatas = []
    for i in range(0, num):
        metadatas.append(MetaData(scene, i))
    return metadatas

end_cutoff = 1
def create_meta_data_list(config, start_cutoff=5):
    global end_cutoff
    shuffle = config['dataset']['shuffle_metadata']
    # shuffle_loader = config['dataset']['shuffle_loader']
    train_list = []
    test_lists = []
    valid_list = []
    batch_size = config['train_parameter']['batch_size']
    num_gpu = config['num_gpu']
    is_block = config['dataset']['is_block']
    if is_block:
        is_block_part = config['dataset']['is_block_part']
    else:
        is_block_part = False
    # vbs = 1
    if is_block:
        block_size = config['dataset']['block_size']
    else:
        block_size = 0

    if "sep" in config['dataset']['mode']:
        train_scenes = list(config['dataset']['train_scene'])
        assert len(train_scenes) > 0, f'config["dataset"]["train_scene"] must be no empty!,\n\
config["dataset"]:{dict_to_string(convert_to_dict(config["dataset"]))}'
        for item in train_scenes:
            dir_name = item['name']
            path = config['job_config']['dataset_path'][item['config'].get('path_alias', 'default')]
            res = glob.glob(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            assert len(res) > 0, f"{config['buffer_config']['basic_part_enable_list'][0]} in {path}/{dir_name} not found.\
\n(path_alias: {item['config'].get('path_alias', 'default')}) in config: {item['config']}"
            log.debug(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            log.debug(path)
            log.debug(dir_name)
            log.debug(len(res))
            num = len(res) - start_cutoff - end_cutoff
            index = np.arange(start_cutoff, start_cutoff + num)
            sep_rule = item['config'].get('indice', [])
            if len(sep_rule) == 1:
                num = min(sep_rule[0], num)
                index = np.arange(start_cutoff, start_cutoff + num)
            elif len(sep_rule) == 2:
                start = sep_rule[0]
                end = min(num, sep_rule[1])
                num = end - start
                index = np.arange(start_cutoff + start, start_cutoff + end)

            if is_block:
                index = index[:-block_size - 1:block_size]
                num = len(index)

            # scale = item['config'].get('scale', 1)

            # if scale!=1:
            #     np.random.seed(time.time())
            #     index = np.random.choice(index, int(len(list(index)) * scale), replace=False)
            #     num = len(index)

            train_list += [MetaDataWithPath(dir_name, index[i], item['config'].get('path_alias', 'default'))
                           for i in range(num)]

            log.info("train_scene: {}, path: {} len: {}".format(
                dir_name,
                path,
                num))

        test_scenes = list(config['dataset']['test_scene'])
        for item in test_scenes:
            dir_name = item['name']
            path = config['job_config']['dataset_path'][item['config'].get('path_alias', 'default')]
            res = glob.glob(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            assert len(res) > 0, f"{config['buffer_config']['basic_part_enable_list'][0]} in {path}/{dir_name} not found.\nconfig: {item}"
            # log.debug(path)
            # log.debug(dir_name)
            # log.debug(res)
            num = len(res) - start_cutoff - end_cutoff
            index = np.arange(start_cutoff, start_cutoff + num)
            sep_rule = item['config'].get('indice', [])
            if len(sep_rule) == 1:
                end = sep_rule[0]
                index = index[:end]
            elif len(sep_rule) == 2:
                start = sep_rule[0]
                end = sep_rule[1]
                index = index[start:end]
            if is_block:
                if not is_block_part:
                    index = index[:-block_size+1:block_size]
            num = len(index)
            test_lists.append([MetaDataWithPath(dir_name, index[i], item['config'].get('path_alias', 'default'))
                          for i in range(num)])
            log.info("test_scene: {}, path: {} len: {}".format(
                dir_name,
                path,
                num))
        # log.debug(dict_to_string(test_lists[0]))
        # exit()
    else:
        raise NotImplementedError(
            f"create dataset with {config['dataset']['mode']} mode, but only \'seq\' mode supported for dataset!")

    is_initial_shuffle_metadata = True
    if is_initial_shuffle_metadata:
        random.seed(2025)
        random.shuffle(train_list)
        
    if shuffle:
        random.seed(time.time())
        random.shuffle(train_list)
    
    train_scale = config["dataset"].get("train_scale", 1)

    if train_scale != 1:
        log.debug(f"train_scale={train_scale}, scaling train_list(len={len(train_list)})")
        np.random.seed(2025)
        train_ind = np.random.choice(np.arange(len(train_list), dtype=int), int(len(train_list) * train_scale), replace=False)
        train_list = list(np.array(train_list)[train_ind])
        log.debug(f"scaled train_list(len={len(train_list)})")

    if is_block:
        minimum_total_size = num_gpu * batch_size
        while len(train_list) % (minimum_total_size) != 0:
            train_list += train_list[:minimum_total_size - len(train_list) % minimum_total_size]
        
        if is_block_part:
            def generate_block_metadata(block_list: list[MetaDataWithPath], _batch_size, _num_gpu, _block_size, round_block=False):
                part_size = config['dataset']['part_size']
                assert _block_size % part_size == 0 
                _minimum_total_size = _num_gpu * _batch_size
                assert len(block_list) % _minimum_total_size == 0
                ''' expand_list: expand block_list with part_size '''
                expand_list = []
                for md in block_list:
                    expand_list.append(md)
                    for block_id in range(part_size, _block_size, part_size):
                        expand_list.append(md.get_offset(block_id))
                ''' len_expend_list = num_parts '''
                len_expand_list = len(expand_list) 
                _num_part_per_block = _block_size // part_size
                # log.debug(f'{len(block_list)} {len_expand_list} {len(block_list) // _minimum_total_size * _minimum_total_size * parted_block_size}')
                assert len_expand_list == len(block_list) * _num_part_per_block
                ret_list = []
                len_batched_seq = _batch_size * _num_part_per_block
                for seq_id in range(len_expand_list // len_batched_seq):
                    cut_list = expand_list[seq_id * len_batched_seq: (seq_id+1) * len_batched_seq]
                    for block_id in range(_num_part_per_block):
                        for batch_id in range(0, _batch_size):
                            ret_list.append(cut_list[batch_id * _num_part_per_block + block_id])
                return ret_list
            train_list = generate_block_metadata(train_list, batch_size, num_gpu,block_size)
            # valid_list = generate_block_metadata(valid_list, 1, 1, block_size)
            # test_list = generate_block_metadata(test_list, 1, 1, block_size)
    else:
        minimum_total_size = batch_size * num_gpu
        while len(train_list) % (minimum_total_size) != 0:
            train_list += train_list[:minimum_total_size - len(train_list) % minimum_total_size]
    log.debug("train: {} ... {} len={}".format(str(train_list[:3]), str(train_list[-3:]), len(train_list)))
    log.debug("test: {} ... {} len={}".format(str(test_lists[0][:3]), str(test_lists[0][-3:]), len(test_lists)))
    log.info("complete creating metadata.")
    return train_list, valid_list, test_lists


''' a single frame meta info, not including data. '''
''' in raw_data_importer, index is the frame index. '''
''' in patch_loader, index is the npz index, (= frame_index - start_offset) '''


class CropMetaData(MetaData):
    def __init__(self, scene_name, index, global_index, skybox_ratio, discontinuity_ratio):
        super().__init__(scene_name, index)
        # index is frame_index, global_index is cropped patch index
        self.global_index = global_index
        self.skybox_ratio = skybox_ratio
        self.discontinuity_ratio = discontinuity_ratio

    def to_dict(self):
        return {'scene_name': self.scene_name,
                'index': self.index,
                'global_index': self.global_index,
                'skybox_ratio': self.skybox_ratio,
                'discontinuity_ratio': self.discontinuity_ratio}


class DatasetBase(Dataset):
    def __init__(self, dataset_name, metadatas: list[MetaDataWithPath], mode="train"):
        self.dataset_name = dataset_name
        self.metadatas = metadatas
        self.mode = mode
        log.info("dataset_name: {}, data_size: {}".format(
            self.dataset_name, self.__len__()))

    @staticmethod
    def preprocess(data, config={}):
        ret = {}
        for name in data.keys():
            # if ('world_position' in name):
            #     scale_factor = 2.0
            #     data[name] = torch.clamp(
            #         data[name], -65536.0 * scale_factor, 65536.0 * scale_factor)
            #     data[name] = data[name] / 65536.0 / scale_factor
            if isinstance(data[name], torch.Tensor) and ('scene_color' in name or 'sky_color' in name or 'st_color' in name):
                ret[name] = tonemap_func(data[name], use_global_settings=True, mean_map=DatasetGlobalConfig.log_tonemapper__color_mean_map)
            elif isinstance(data[name], torch.Tensor) and ('scene_light' in name):
                ret[name] = tonemap_func(data[name], use_global_settings=True, mean_map=DatasetGlobalConfig.log_tonemapper__light_mean_map)
            elif 'normal' in name:
                ret[name] = data[name] * 0.5 + 0.5
            else:
                ret[name] = data[name]
        return ret

    def __len__(self) -> int:
        return len(self.metadatas)
