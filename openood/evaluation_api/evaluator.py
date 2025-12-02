from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import seaborn as sns
from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor


class Evaluator:
    def __init__(
            self,
            net: nn.Module,
            id_name: str,
            data_root: str = './data',
            config_root: str = './configs',
            preprocessor: Callable = None,
            postprocessor_name: str = None,
            postprocessor: Type[BasePostprocessor] = None,
            batch_size: int = 200,
            shuffle: bool = False,
            num_workers: int = 8,
            fsood: bool = False,
            cached_dir: str = "./cache",
            use_cache: bool = False,
            **postpc_kwargs
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name,
                                              id_name)
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError(
                'postprocessor should inherit BasePostprocessor in OpenOOD')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor,
                                                fsood=fsood,
                                                cached_dir=cached_dir,
                                                use_cache=use_cache,
                                                **loader_kwargs)

        # wrap base model to work with certain postprocessors
        if postprocessor_name == 'react':
            net = ReactNet(net)
        elif postprocessor_name == 'ash':
            net = ASHNet(net)

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'],
                            use_cache=use_cache, id_name=id_name, **postpc_kwargs)

        # postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'],
                            # **postpc_kwargs)
        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.postprocessor_name = postprocessor_name
        self.postpc_kwargs = postpc_kwargs
        self.metrics, self.scores = self.init_dicts()
        self.iddata = dataloader_dict['id']
        # perform hyperparameter search if have not done so
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.hyperparam_search()

        self.net.eval()

        # how to ensure the postprocessors can work with
        # models whose definition doesn't align with OpenOOD

    def init_dicts(self):
        metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in self.dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                    {k: None
                     for k in self.dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in self.dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in self.dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in self.dataloader_dict['csid'].keys()},
        }
        return metrics, scores

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, data_name: str = 'id') -> float:
        if data_name == 'id':
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        elif data_name == 'csid':
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc
        else:
            raise ValueError(f'Unknown data name {data_name}')

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'
        if self.metrics[task] is None:
            self.net.eval()

            # id score
            if self.scores['id']['test'] is None:   # load score 
                print(f'Performing inference on {self.id_name} test set...',
                      flush=True)
                id_pred, id_conf, id_gt, id_feat = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                self.scores['id']['test'] = [id_pred, id_conf, id_gt, id_feat]   # mstar test 
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']

            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'Performing inference on {self.id_name} '
                            f'(cs) test set [{i + 1}]: {dataset_name}...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = \
                            self.postprocessor.inference(
                                self.net,
                                self.dataloader_dict['csid'][dataset_name],
                                progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # We evaluate InD accuracy only with one propagation
            # self.eval_acc(id_name)
            assert len(id_pred) == len(id_gt)
            correct = (id_pred == id_gt).sum().item()
            acc = correct / len(id_gt) * 100
            self.metrics['id_acc'] = acc
            print("InD ACC:", acc)

            # load nearood data and compute ood metrics
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt, id_feat],
                                          ood_split='near',
                                          progress=progress)
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt, id_feat],
                                         ood_split='far',
                                         progress=progress)

            near_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                           len(near_metrics))
            far_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                          len(far_metrics))

            self.metrics[task] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) +
                      ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) +
                      ['farood'],
                columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC'],
            )
        else:
            print('Evaluation has already been done!')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # more options can be specified also
            print(self.metrics[task])

        return self.metrics[task]

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True,
                  save: bool = True):
        print(f'Processing {ood_split} ood...', flush=True)
        [id_pred, id_conf, id_gt, id_feat] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][
            ood_split].items():

            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                ood_pred, ood_conf, ood_gt, ood_feat = self.postprocessor.inference(
                    self.net, ood_dl, progress)
                self.scores['ood'][ood_split][dataset_name] = [
                    ood_pred, ood_conf, ood_gt
                ]
            else:
                print(
                    'Inference has been performed on '
                    f'{dataset_name} dataset...',
                    flush=True)
                [ood_pred, ood_conf,
                 ood_gt] = self.scores['ood'][ood_split][dataset_name]

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            feature = np.concatenate((id_feat, ood_feat), axis=0)

            if save:
                dir = f"ood_results/{self.postprocessor_name}_{self.id_name}"
                if self.postprocessor_name == "nac":
                    sub_dir = ""
                    for ln, ln_hypers in self.postprocessor.layer_kwargs.items():
                        ln_prefix = [f"{k}_{v}" for k, v in ln_hypers.items() if k == 'method']
                        sub_dir += f"_{self.postprocessor.ln_to_aka[ln]}_" + "_".join(ln_prefix)
                    dir = f"{dir}/{sub_dir}"
                os.makedirs(dir, exist_ok=True)
                np.save(f"{dir}/in_scores.npy", id_conf)
                np.save(f"{dir}/{dataset_name}_out_scores.npy", ood_conf)

            id_conf = conf[label != -1]  # 取ID样本的置信度
            ood_conf = conf[label == -1]  # 取OOD样本的置信度

            # print_KDE_plt(id_conf, ood_conf, dataset_name, self.postprocessor_name)
            if dataset_name == 'mstar7':
                id_dataloader = self.dataloader_dict['id']['test']
                ood_dataloader = self.dataloader_dict['ood']['far']['mstar7']
                combined_dataloader = merge_dataloaders(id_dataloader, ood_dataloader)
                idood_thresholds(id_pred, id_gt, ood_conf, ood_pred, ood_gt, dataset_name, id_conf, 
                                self.net, combined_dataloader, self.postprocessor)
            elif dataset_name == 'SAMPLE':
                id_dataloader = self.dataloader_dict['id']['test']
                ood_dataloader = self.dataloader_dict['ood']['near']['SAMPLE']
                combined_dataloader = merge_dataloaders(id_dataloader, ood_dataloader)
                idood_thresholds(id_pred, id_gt, ood_conf, ood_pred, ood_gt, dataset_name, id_conf, 
                            self.net, combined_dataloader, self.postprocessor)
            elif dataset_name == 'SAMPLE':
                id_dataloader = self.dataloader_dict['id']['test']
                ood_dataloader = self.dataloader_dict['ood']['far']['SAMPLE']
                combined_dataloader = merge_dataloaders(id_dataloader, ood_dataloader)
                idood_thresholds(id_pred, id_gt, ood_conf, ood_pred, ood_gt, dataset_name, id_conf, 
                            self.net, combined_dataloader, self.postprocessor)
            elif dataset_name == 'SHIP':
                id_dataloader = self.dataloader_dict['id']['test']
                ood_dataloader = self.dataloader_dict['ood']['far']['SHIP']
                combined_dataloader = merge_dataloaders(id_dataloader, ood_dataloader)
                idood_thresholds(id_pred, id_gt, ood_conf, ood_pred, ood_gt, dataset_name, id_conf, 
                            self.net, combined_dataloader, self.postprocessor)

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        self._print_metrics(list(metrics_mean[0]))

        
        df = pd.DataFrame({'Confidence': np.concatenate([id_conf, ood_conf]),
                   'Sample Type': ['ID'] * len(id_conf) + ['OOD'] * len(ood_conf)})
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

                # tsne_ood_gt = 3 * np.ones_like(ood_gt)
                # tsne_label = np.concatenate([id_gt, tsne_ood_gt])
                # t_sne_show(feature, tsne_label, 4, 
                #            '/home/hhq/ood_coverage-master/openood/evaluation_api/plt/tsne_MSTAR7.png')

    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out, _] = metrics
        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
            flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        print("Sweep over {} number of combinations".format(len(hyperparam_combination)))
        final_index = None

        for i, hyperparam in tqdm(enumerate(hyperparam_combination)):
            self.postprocessor.set_hyperparam(hyperparam)
            if self.postprocessor_name == 'nac':
                self.postprocessor.build_nac(self.net, self.iddata, reload=True, prefix=self.id_name)

            id_pred, id_conf, id_gt, _ = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'], progress=False)
            ood_pred, ood_conf, ood_gt, _ = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'], progress=False)

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            fpr, auroc = ood_metrics[0], ood_metrics[1]
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

            # Sub-Train-Evaluation for sanity check
            # we calculate fpr & auroc between InD training and validation set
            ood_pred, ood_conf, ood_gt, _ = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['sub_train'], progress=False)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            train_fpr, train_auroc = ood_metrics[0], ood_metrics[1]

            print('Hyperparam: {}, fpr: {}, auroc: {}, '
                  'train-fpr: {}, train-auroc: {}'.format(hyperparam, fpr, auroc, train_fpr, train_auroc))

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
            self.postprocessor.get_hyperparam()))
        if self.postprocessor_name == 'nac':
            self.postprocessor.build_nac(self.net, reload=True, prefix=self.id_name)
            self.metrics, self.scores = self.init_dicts()

        self.postprocessor.hyperparam_search_done = True

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results

from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
sns.set_style('whitegrid') 
sns.set_palette('muted')
RS = 600

def unique_labels(y_true, y_pred):
    labels = np.concatenate((y_true, y_pred), axis=0)
    return np.unique(labels)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Oranges):
    """
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
    # Compute confusion matrix
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    # print(cm.shape[1])
    # print(cm.shape[0])
    font = FontProperties(family='Times New Roman', size=12)
    # print(len(y_pred),len(y_true))
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    # classes = list(set(y_true)&set(y_pred))
    # print(classes,len(classes))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    else:
        pass

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    im.set_clim(0, 1200)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label'
           )
    ax.set_xlabel('Predicted label', fontproperties='Times New Roman', fontsize=14)
    ax.set_ylabel('True label', fontproperties='Times New Roman', fontsize=14)
    ax.set_ylim(len(classes) - 0.5, -0.5)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font)

    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")
    plt.grid(False)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
 
def idood_thresholds(id_pred, id_gt, ood_conf, ood_pred, ood_gt, dataset_name, id_conf, model, dataloader,
                     postprocessor):

    label = np.concatenate([id_gt, -1 * np.ones_like(ood_gt)])
    conf = np.concatenate([id_conf, ood_conf])
    pred = np.concatenate([id_pred, ood_pred])

    thresholds = np.linspace(-20, 10, 1001)  
    best_threshold, best_accuracy = 0, 0  
    for threshold in thresholds:
        is_ood = conf < threshold
        temp_predictions = np.where(is_ood, -1, pred) 
        
        correct_predictions = (temp_predictions == label)  
        overall_accuracy = correct_predictions.sum() / len(label) 

        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_threshold = threshold

    print(f"Selected Best Threshold: {best_threshold}, Best Overall Accuracy: {best_accuracy}")


    # is_ood = conf <= best_threshold - 0.58 # - 0.58
    # is_ood = conf <=  -10.19 - 0.58
    # is_ood = conf < 0.39
    final_predictions = np.where(is_ood, -1, pred)


    id_indices = np.where(~is_ood)[0]  
    id_inputs = [dataloader.dataset[i] for i in id_indices] 

    model.eval()
    new_id_preds = []
    with torch.no_grad():
        for inputs, _ in id_inputs:  
            outputs = model(inputs.unsqueeze(0).cuda()) 
            # outputss = postprocessor.in
            new_id_preds.append(outputs.argmax(dim=1).item())
    new_id_preds = np.array(new_id_preds)

    final_predictions[id_indices] = new_id_preds

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    precision = precision_score(label, final_predictions, average='macro', zero_division=0)
    recall = recall_score(label, final_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(label, final_predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(label, final_predictions)

    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


    class_names = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9","10", "-1"])  
    plot_confusion_matrix(label, final_predictions, classes=class_names, normalize=False)
    plt.savefig(
        f'/home/hhq/SAROOD/openood/evaluation_api/plt/confusion_{dataset_name}_MSTAR.png',
        format='png',
        dpi=500,
        bbox_inches='tight'
    )



def scatter(x, colors, class_num, save_path):
    # We choose a color palette with seaborn.
    color_map = ['red', 'lightskyblue', 'yellow', 'gray', 'green', 'mediumpurple', 'black', 'orange',
                 'pink', 'cyan', 'brown']

    # We create a scatter plot.
    plt.figure(figsize=(8, 8))
    labels = ['2S1', 'BMP2', 'BRDM_2', 'BTR60', 'BTR70', 'D7', 'unknown', 'T62','T72',
              'ZIL131','ZSU_23_4']
    for i in range(class_num):
        ind = np.where(colors == i)[0]
        plt.scatter(x[ind,0], x[ind,1], lw=0, s=40, color=color_map[i], label=labels[i], alpha=1)
    # for i in range(10, class_num+10):
    #     ind = np.where(colors == i)[0]
    #     plt.scatter(x[ind,0], x[ind,1], lw=1, s=200, color=color_map[i-10], edgecolors='black', marker='*',
    #                 label='class center of '+labels[i-10], alpha=1)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.grid(False)
    # plt.legend(loc='upper left', prop={'family':'Times New Roman', 'size':12}, ncol=1, fontsize=20)
    plt.axis('off')
    plt.axis('tight')
    plt.savefig(save_path, dpi=200)


def t_sne_show(logits, test_labels_new, class_num, save_path):
    X = logits
    y = test_labels_new
    digits_proj = TSNE(n_iter=2000, init='random', learning_rate=500, random_state=RS, perplexity=50).fit_transform(X)
    scatter(digits_proj, y, class_num, save_path)


def print_KDE_plt(id_conf, ood_conf, dataset_name, method):


    plt.figure(figsize=(10, 6))
    sns.kdeplot(id_conf, shade=True, color='blue', label='ID Samples', linewidth=2)
    sns.kdeplot(ood_conf, shade=True, color='red', label='OOD Samples', linewidth=2)       

    plt.xlabel('Confidence Scores', fontsize=12)
            
    plt.ylabel('Density', fontsize=12)
    plt.title('ID vs OOD Confidence Score Distribution (MSTAR vs {})'.format(dataset_name), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('/home/hhq/SAROOD/openood/evaluation_api/plt/KDE_{}_{}_MSTAR.png'.format(dataset_name, method), format='png', dpi=1000,
                    bbox_inches='tight')
    


from torch.utils.data import DataLoader, TensorDataset
import torch

def merge_dataloaders(id_dataloader, ood_dataloader):

    id_data, id_labels = [], []
    for batch in id_dataloader:
        id_data.append(batch['data'])       
        id_labels.append(batch['label'])   
    id_data = torch.cat(id_data, dim=0)
    id_labels = torch.cat(id_labels, dim=0)


    ood_data, ood_labels = [], []
    for batch in ood_dataloader:
        ood_data.append(batch['data'])       
    ood_data = torch.cat(ood_data, dim=0)
    ood_labels = -1 * torch.ones(ood_data.size(0), dtype=torch.long)  


    combined_data = torch.cat([id_data, ood_data], dim=0)
    combined_labels = torch.cat([id_labels, ood_labels], dim=0)


    combined_dataset = TensorDataset(combined_data, combined_labels)
    combined_dataloader = DataLoader(combined_dataset, batch_size=64, shuffle=False)

    return combined_dataloader
