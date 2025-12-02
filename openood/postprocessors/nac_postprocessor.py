from typing import Any
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import faiss
from .base_postprocessor import BasePostprocessor

from .nac.utils import StatePooling, TrainSubset
from .nac.instr_state import get_intr_name
from .nac.coverage import make_layer_size_dict, KMNC

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class NACPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NACPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.test_method = self.args.test_method
        self.layer_names = self.args.layer_names  # [avgpool, layer3, layer2, layer1]
        self.valid_num = self.args.valid_num

        self.layer_kwargs = self.config.postprocessor.layer_kwargs
        self.args_dict = self.config.postprocessor.postprocessor_sweep  # search for hyperparameters 

        self.model_name = None
        self.spatial_func = None
        self.Coverage = None
        self.build_nac_flag = True if not self.config.postprocessor.APS_mode else False
        self.ln_to_aka, self.aka_to_ln = None, None

        self.unique_id = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
        self.save_dir = f"./coverage_cache/{self.unique_id}"
        self.setup_flag = False

    def load(self, f_name, unpack, layer_features, params=None):
        self.Coverage = KMNC.load(self.save_dir, f_name, layer_features, unpack=unpack,
                                  params=params, verbose=False)

    def save(self, prefix="imagenet"):
        os.makedirs(self.save_dir, exist_ok=True)
        self.Coverage.save(self.save_dir, prefix=prefix,
                           aka_ln=self.ln_to_aka, verbose=False)

    def build_nac(self, net: nn.Module, id_loader_dict, reload=False, prefix="imagenet"):
        self.setup_flag = False
        activation_log = []
        if not self.setup_flag:
            layer_features = {ln: [] for ln in self.layer_names}
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature_list = net(data, return_feature_list=True)
                    _, feature_knn = net(data, return_feature=True)
                    for ln, feature in zip(self.layer_names, feature_list):
                        layer_features[ln].append(feature.cpu().numpy())  # 转到 CPU 存储
            # self.setup_flag = True

                    activation_log.append(
                        normalizer(feature_knn.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(feature_knn.shape[1])
            self.index.add(self.activation_log)
        else:
            pass

        # layer_features = layer_features
        unpack = lambda b, device: (b['data'].to(device), b['label'].to(device))
        f_name = prefix
        for ln, ln_hypers in self.layer_kwargs.items():
            ln_prefix = [f"{k}_{v}" for k, v in ln_hypers.items() if k != 'O']
            f_name += f"_{self.ln_to_aka[ln]}_" + "_".join(ln_prefix)
        f_name += f"_states.pkl"

        reload = False
        if reload and os.path.isfile(os.path.join(self.save_dir, f_name)):
            self.load(f_name, unpack, layer_features, params=self.layer_kwargs)
        else:
            self.Coverage = KMNC(self.layer_size_dict, layer_features,
                                 hyper=self.layer_kwargs,
                                 unpack=unpack)   # init_variable 
            if self.use_cache:
                self.Coverage.assess_with_cache(self.nac_dataloader)
            else:
                self.Coverage.assess(net, self.nac_dataloader, self.index,
                                     self.spatial_func)
            self.save(prefix=prefix)
        self.Coverage.update()
        self.build_nac_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, id_name="imagenet",
              valid_num=None, layer_names=None, aps=None, use_cache=False, **kwargs):
        self.setup_flag = False
        self.model_name = net.__class__.__name__
        if hasattr(net, "backbone"):
            self.model_name = net.backbone.__class__.__name__
            print(self.model_name)
        if valid_num is not None:
            self.valid_num = valid_num  # 1000
        if layer_names is not None:
            self.layer_names = layer_names  # ['avgpool', 'layer3', 'layer2', 'layer1']
        if aps is not None:
            self.APS_mode = aps  # False
            self.build_nac_flag = not aps  # True
        self.use_cache = use_cache
        self.layer_kwargs = {ln: self.layer_kwargs[ln] for ln in self.layer_kwargs if ln in self.layer_names} # 50,500,50,500
        self.args_dict = {f"{ln}_{k}": v for ln in self.layer_kwargs for k, v in self.args_dict[ln].items()}

        self.aka_to_ln, self.layer_names = get_intr_name(self.layer_names, self.model_name, net)
        self.ln_to_aka = {v: k for k, v in self.aka_to_ln.items()}
        self.layer_kwargs = {self.aka_to_ln[aka]: v for aka, v in self.layer_kwargs.items()}
        print(f"Setup NAC Postprocessor (valid_num:{self.valid_num}, layers:{self.layer_names})......")

        self.spatial_func = StatePooling(self.model_name)  # avgpool
        if self.use_cache:   # false 
            self.nac_dataloader = id_loader_dict['main_train']
            dummy_shape = (3, 32, 32) if "cifar" in id_name else (3, 224, 224)
        else:
            self.nac_dataset = TrainSubset(id_loader_dict['main_train'].dataset,
                                           valid_num=self.valid_num,
                                           balanced=True)
            self.nac_dataloader = DataLoader(self.nac_dataset,
                                             batch_size=64, shuffle=False,
                                             num_workers=8, pin_memory=True,
                                             drop_last=False)

            dummy_shape = self.nac_dataset.dataset[0]['data'].shape  # (3, 64,64 )
        print("Input shape:", dummy_shape)
        self.layer_size_dict = make_layer_size_dict(net, self.layer_names,
                                                    input_shape=(3, *dummy_shape),
                                                    spatial_func=self.spatial_func)
        print(self.layer_size_dict)
        if self.build_nac_flag:
            self.build_nac(net, id_loader_dict)

    def inference(self, net, data_loader, progress=True):
        if self.use_cache:
            confs, flags, preds, labels = self.Coverage.assess_ood_with_cache(data_loader,
                                                                              progress=progress)
        else:
            confs, flags, preds, labels, features = self.Coverage.assess_ood(net, data_loader,
                                                                             self.index,
                                                                   self.spatial_func,
                                                                       # avgpool, output batch, channels
                                                                   progress=progress)
        return preds, confs, labels, features



    def set_hyperparam(self, hyperparam: list):
        assert (len(hyperparam) / 4) == len(self.layer_kwargs)
        print("##" * 30)
        i = 0
        for ln in self.layer_kwargs:
            O, M, sig_alpha, method = hyperparam[i:i + 4]
            self.layer_kwargs[ln].update({"O": O, "M": M, "sig_alpha": sig_alpha, "method": method})
            print("Set {} paramters to O:{}, M:{}, sig_alpha:{}, method:{}".format(ln, O, M, sig_alpha, method))
            i = i + 4
        self.build_nac_flag = True

    def get_hyperparam(self):
        print_str = ""
        for ln in self.layer_kwargs:
            print_str += "\n{} paramters O:{}, M:{}, " \
                         "sig_alpha:{}, method:{}".format(ln, *list(self.layer_kwargs[ln].values()))
        return print_str
