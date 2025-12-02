import os
import gdown
import zipfile

from torch.utils.data import DataLoader
import torchvision as tvs

if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor

from .preprocessor import get_default_preprocessor, ImageNetCPreProcessor

# csid covariate shift, training data are nuture, csid data are add some noise or other 

DATA_INFO = {
    'mstar': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/train_mstar.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/val_mstar.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/test_mstar.txt'
            },
            'sub_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/sub_train_mstar.txt'
            },
            'main_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/main_train_mstar.txt'
            },
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/test_ship.txt'
            },
            'near': {
                'datasets': ['SAMPLE'],
                'SAMPLE': {
                    'data_dir': 'images_classic/',
                    'imglist_path':'benchmark_imglist/mstar/test_sample.txt'
                },
            },
            'far': {
                'datasets': ['AIRPLANE', 'SHIP'],
                'AIRPLANE': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/mstar/test_airplane.txt'},
                'SHIP': {
                    'data_dir': 'images_classic/',
                    'imglist_path':'benchmark_imglist/mstar/test_ship.txt'
                },
            }
        }
    },
    'mstar3': {
        'num_classes': 3,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/osr_mstar3/train/train_mstar10_3_seed1.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/osr_mstar3/val/val_mstar10_3_seed1.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/osr_mstar3/test/test_mstar10_3_id_seed1.txt'
            },
            'sub_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/sub_train_mstar3_osr.txt'
            },
            'main_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/main_train_mstar3_osr.txt'
            },
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/mstar/test_ship.txt'
            },
            'near': {
                'datasets': ['mstar7'],
                'mstar7': {
                    'data_dir': 'images_classic/',
                    'imglist_path':'benchmark_imglist/osr_mstar3/test/test_mstar10_7_ood_seed1.txt'
                },
            },
            'far': {
                'datasets': ['mstar7'],
                'mstar7': {
                    'data_dir': 'images_classic/',
                    'imglist_path':'benchmark_imglist/osr_mstar3/test/test_mstar10_7_ood_seed1.txt'
                },
            }
        }
    },
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            },
            'sub_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/sub_train_cifar10.txt'
            },
            'main_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/main_train_cifar10.txt'
            },
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
            },
            'near': {
                'datasets': ['mnist'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/cifar10/test_mnist.txt'
                },
            },
            'far': {
                'datasets': ['mstar'],
                'mstar': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mstar.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            },
            'sub_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/sub_train_cifar100.txt'
            },
            'main_train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/main_train_cifar100.txt'
            },

        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                    'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                    'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet200/test_imagenet200.txt'
            },
            'sub_train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet200/sub_train_imagenet200.txt'
            },
            'main_train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet200/main_train_imagenet200.txt'
            },

        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                    'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                    'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                    'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                        'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            },
            'sub_train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/sub_train_imagenet.txt'
            },
            'main_train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/main_train_imagenet.txt'
            },

        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}

download_id_dict = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'benchmark_imglist': '1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP'
}

dir_dict = {
    'images_classic/': [
        'cifar100', 'tin', 'tin597', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist', 'mstar',
    ],
    'images_largescale/': [
        'imagenet_1k',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'places',
        'sun',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r',
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

benchmarks_dict = {
    'cifar10':
        ['cifar10','mnist', 'mstar'],
    'mstar':
        ['sarship'],
    'mstar3':
        ['mstar7'],
    'cifar100':
        ['cifar100', 'cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'imagenet': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
}


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, data_root):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(data_root, key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


def data_setup(data_root, id_data_name):
    if not data_root.endswith('/'):
        data_root = data_root + '/'

    if not os.path.exists(os.path.join(data_root, 'benchmark_imglist')):
        gdown.download(id=download_id_dict['benchmark_imglist'],
                       output=data_root)
        file_path = os.path.join(data_root, 'benchmark_imglist.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(data_root)
        os.remove(file_path)

    for dataset in benchmarks_dict[id_data_name]:
        download_dataset(dataset, data_root)


from openood.postprocessors.nac.utils import load_statistic


def craft_loader(fname, cache_dir="./cache/imagenet/vit-b-16", batch_size=128, **kwargs):
    data_dict = {}
    cached_states = load_statistic(cache_dir, fname)
    for key in cached_states:
        try:
            if len(cached_states[key]) == 3:
                raw_state, flags, kl_state = cached_states[key]
                loader_len = int(len(raw_state) / batch_size) if len(raw_state) > batch_size else 1
                data_dict[key] = list(zip(raw_state.chunk(loader_len), flags.chunk(loader_len),
                                          kl_state.chunk(loader_len)))
            else:
                raw_state, flags = cached_states[key]
                loader_len = int(len(raw_state) / batch_size) if len(raw_state) > batch_size else 1
                data_dict[key] = list(zip(raw_state.chunk(loader_len), flags.chunk(loader_len),
                                          [None] * loader_len))
            # print(len(raw_state), loader_len)
        except:
            data_dict[key] = cached_states[key]
    return data_dict


def get_id_ood_dataloader(id_name, data_root, preprocessor,
                          fsood=False, use_cache=False,
                          cached_dir="./cache/imagenet/vit-b-16",
                          **loader_kwargs):
    if 'imagenet' in id_name:
        if tvs_new:
            if isinstance(preprocessor,
                          tvs.transforms._presets.ImageClassification):
                mean, std = preprocessor.mean, preprocessor.std
            elif isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        else:
            if isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        imagenet_c_preprocessor = ImageNetCPreProcessor(mean, std)

    # weak augmentation for data_aux
    test_standard_preprocessor = get_default_preprocessor(id_name)

    dataloader_dict = {}
    data_info = DATA_INFO[id_name]

    # id
    sub_dataloader_dict = {}
    for split in data_info['id'].keys():
        if split == 'train':
                dataset = ImglistDataset(
                name='_'.join((id_name, split)),
                imglist_pth=os.path.join(data_root,
                                         data_info['id'][split]['imglist_path']),
                data_dir=os.path.join(data_root,
                                      data_info['id'][split]['data_dir']),
                num_classes=data_info['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor)
        if use_cache:
            dataloader = craft_loader(f"{id_name}_id_{split}", cached_dir, **loader_kwargs)
        else:
            dataset = ImglistDataset(
                name='_'.join((id_name, split)),
                imglist_pth=os.path.join(data_root,
                                         data_info['id'][split]['imglist_path']),
                data_dir=os.path.join(data_root,
                                      data_info['id'][split]['data_dir']),
                num_classes=data_info['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor)
            dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[split] = dataloader
    dataloader_dict['id'] = sub_dataloader_dict

    # csid
    sub_dataloader_dict = {}
    for dataset_name in data_info['csid']['datasets']:
        if fsood:
            if use_cache:
                dataloader = craft_loader(f"{id_name}_csid_{dataset_name}",
                                          cached_dir, **loader_kwargs)
            else:
                dataset = ImglistDataset(
                    name='_'.join((id_name, 'csid', dataset_name)),
                    imglist_pth=os.path.join(
                        data_root, data_info['csid'][dataset_name]['imglist_path']),
                    data_dir=os.path.join(data_root,
                                          data_info['csid'][dataset_name]['data_dir']),
                    num_classes=data_info['num_classes'],
                    preprocessor=preprocessor
                    if dataset_name != 'imagenet_c' else imagenet_c_preprocessor,
                    data_aux_preprocessor=test_standard_preprocessor)
                dataloader = DataLoader(dataset, **loader_kwargs)
            sub_dataloader_dict[dataset_name] = dataloader
    dataloader_dict['csid'] = sub_dataloader_dict

    # ood
    dataloader_dict['ood'] = {}
    for split in data_info['ood'].keys():  # val near far 
        split_config = data_info['ood'][split]
        print('split_config', split_config)
        if split == 'val':
            # validation set
            if use_cache:
                dataloader = craft_loader(f"{id_name}_ood_val", cached_dir, **loader_kwargs)
            else:
                print(data_root)
                dataset = ImglistDataset(
                    name='_'.join((id_name, 'ood', split)),
                    imglist_pth=os.path.join(data_root,
                                             split_config['imglist_path']),
                    data_dir=os.path.join(data_root, split_config['data_dir']),
                    num_classes=data_info['num_classes'],
                    preprocessor=preprocessor,
                    data_aux_preprocessor=test_standard_preprocessor)
                dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['ood'][split] = dataloader
        else:
            # dataloaders for nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config['datasets']: # mstar
                if use_cache:
                    dataloader = craft_loader(f"{id_name}_ood_{split}_{dataset_name}",
                                              cached_dir, **loader_kwargs)
                else:
                    dataset_config = split_config[dataset_name]
                    dataset = ImglistDataset(
                        name='_'.join((id_name, 'ood', dataset_name)),
                        imglist_pth=os.path.join(data_root,
                                                 dataset_config['imglist_path']),
                        data_dir=os.path.join(data_root,
                                              dataset_config['data_dir']),
                        num_classes=data_info['num_classes'],
                        preprocessor=preprocessor,
                        data_aux_preprocessor=test_standard_preprocessor)
                    dataloader = DataLoader(dataset, **loader_kwargs)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict['ood'][split] = sub_dataloader_dict

    # print(dataloader_dict)
    return dataloader_dict
