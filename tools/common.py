"""Shared helpers for the MambaCenter tools: paths, YAML experiment configs, CenterNet opts."""
import copy
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CENTERNET_LIB = ROOT / 'baseline' / 'CenterNet' / 'src' / 'lib'


def run_dir(cfg):
    """runs/<name>_s<seed> under $MC_RUNS (cluster: /opt) or the repository's runs/."""
    root = Path(os.environ.get('MC_RUNS', ROOT / 'runs'))
    return root / ('%s_s%d' % (cfg['name'], cfg['seed']))


def setup_paths():
    """Make CenterNet's src/lib and the mambacenter package importable."""
    for path in (str(ROOT), str(CENTERNET_LIB)):
        if path not in sys.path:
            sys.path.insert(0, path)


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, overrides=()):
    """Load an experiment YAML; ``base:`` inherits another file, ``a.b=v`` overrides keys."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop('base', None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError('override must look like key.sub=value: %s' % item)
        node = cfg
        *parents, leaf = key.split('.')
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def centernet_opt(cfg, save_dir, data_dir):
    """CenterNet opts for one experiment config; returns (opt, dataset class)."""
    setup_paths()
    from opts import opts
    from datasets.dataset_factory import get_dataset

    train, loss = cfg['train'], cfg.get('loss', {})
    argv = ['ctdet', '--dataset', cfg.get('dataset', 'visdrone'), '--arch', cfg['arch'],
            '--exp_id', cfg['name'], '--input_res', str(cfg['input_res']),
            '--down_ratio', '4', '--head_conv', str(cfg.get('head_conv', 256)),
            '--K', str(cfg['eval']['K']), '--batch_size', str(train['batch_size']),
            '--lr', repr(float(train['lr'])), '--lr_step', ','.join(map(str, train['lr_step'])),
            '--num_epochs', str(train['epochs']), '--num_workers', str(train['num_workers']),
            '--seed', str(cfg['seed']), '--gpus', '0',
            '--hm_weight', repr(float(loss.get('hm_weight', 1))),
            '--wh_weight', repr(float(loss.get('wh_weight', 0.1))),
            '--off_weight', repr(float(loss.get('off_weight', 1)))]
    argv += [str(a) for a in cfg.get('centernet_args', [])]
    opt = opts().parse(argv)
    dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    opt.data_dir = str(data_dir)
    opt.save_dir = opt.debug_dir = str(save_dir)
    opt.coarse_weight = float(loss.get('coarse_weight', 0))
    opt.coarse_hm = opt.coarse_weight > 0
    opt.pred_weight = float(loss.get('pred_weight', 0))
    return opt, dataset


def create_detector(cfg, opt, pretrained=True):
    """CenterNet create_model with the Mamba architectures registered from cfg['model']."""
    setup_paths()
    from models.model import create_model
    from mambacenter.centernet import register

    model_cfg = copy.deepcopy(cfg.get('model') or {})
    model_cfg['pretrained'] = pretrained
    register(model_cfg)
    return create_model(opt.arch, opt.heads, opt.head_conv)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
