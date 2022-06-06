#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import time
import logging
from tqdm import tqdm
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict

from timm.bits import initialize_device, Tracker, Monitor, AccuracyTopK, AvgTensor
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_transform_v2, create_loader_v2, resolve_data_config, RealLabelsImagenet, \
    PreprocessCfg
from timm.utils import natural_key, setup_default_logging, imagenet_r_mask
from timm.utils import imagenetc_distortions, imagenetc_alexnet_error_rates_list


_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--normalize', action='store_true',
                    help='whether to normalize dataste')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=20, type=int,
                    metavar='N', help='batch logging frequency (default: 20)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# parser.add_argument('--num-gpu', type=int, default=1,
#                     help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--force-cpu', action='store_true', default=False,
                    help='Force CPU to be used even if HW accelerator exists.')

##################
parser.add_argument('--resize_mode', choices=['resize_center_crop', 'resize', 'none'],  default='resize_center_crop',
                    help="select resize_mode for val loader. "
                         "choose reisze_center_crop for most dataset"
                         "choose resize for evaluating Stylized-ImageNet or ImageNet-C with larger resloution (>224)"
                         "choose none to skip crop and resize if inputs are already 224x224 (for evaluating Stylized-ImageNet or ImageNet-C)")
parser.add_argument('--evaluate_imagenet_r', action='store_true', default=False,
                    help="mapping the 1k labels to 200 labels (for evaluate ImageNet-R)")
parser.add_argument('--evaluate_imagenet_c', action='store_true', default=False,
                    help="evaluate 15 distortions times 5 severities in ImageNet-C")

# parser.add_argument('--epochs', type=int,
#                     help='how many epochs the model has been trained')
# parser.add_argument('--max_epochs', type=int,
#                     help='how many epochs the model will be trained in total')
parser.add_argument('--layer_scale_init_value', default=0, type=float,
                    help="Layer scale initial values")
##################

def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp)

    # create model
    try:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            ###########
            layer_scale_init_value=args.layer_scale_init_value,
            ###########
            in_chans=3,
            global_pool=args.gp,
            scriptable=args.torchscript)
    except:
        print('model has no layer_scale_init_value argument...')
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            in_chans=3,
            global_pool=args.gp,
            scriptable=args.torchscript)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    eval_pp_cfg = PreprocessCfg(
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=1.0 if test_time_pool else data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
    )
    # eval_pp_cfg.no_resize = args.no_resize
    eval_pp_cfg.resize_mode = args.resize_mode

    eval_transform = create_transform_v2(cfg=eval_pp_cfg, normalize=args.normalize, is_training=False)


    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    # FIXME device
    model, criterion = dev_env.to_device(model, nn.CrossEntropyLoss())
    model.to(dev_env.device)

    ########################
    if args.evaluate_imagenet_c:
        print("Evaluate ImageNet C")
        assert args.checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)

        error_rates = []
        with open(os.path.join(checkpoint_dir, 'eval_imagenet_c.txt'), 'w') as f:
            for distortion_name in tqdm(imagenetc_distortions):
                error_rate = test_imagenet_c(distortion_name,  model, dev_env, criterion, args, eval_transform, file=f)
                error_rates.append(error_rate)
                print(f'Distortion: {distortion_name}  | CE (unnormalized) (%): {100 * error_rate}')
            print(f'error rates: {error_rates}', file=f, flush=True)
            print(f'mean error rates: {np.mean(error_rates)}', file=f, flush=True)
        return
    else:
        dataset = create_dataset(
            root=args.data, name=args.dataset, split=args.split,
            load_bytes=args.tf_preprocessing, class_map=args.class_map)

        print(f'len(dataset): {len(dataset)}')
        if args.valid_labels:
            with open(args.valid_labels, 'r') as f:
                valid_labels = {int(line.rstrip()) for line in f}
                valid_labels = [i in valid_labels for i in range(args.num_classes)]
        else:
            valid_labels = None

        if args.real_labels:
            real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
        else:
            real_labels = None

        # eval_pp_cfg = PreprocessCfg(
        #     input_size=data_config['input_size'],
        #     interpolation=data_config['interpolation'],
        #     crop_pct=1.0 if test_time_pool else data_config['crop_pct'],
        #     mean=data_config['mean'],
        #     std=data_config['std'],
        # )
        #
        # dataset.transform = create_transform_v2(cfg=eval_pp_cfg, is_training=False)
        dataset.transform = eval_transform

        loader = create_loader_v2(
            dataset,
            batch_size=args.batch_size,
            pp_cfg=eval_pp_cfg,
            num_workers=args.workers,
            pin_memory=args.pin_mem)

        loss_avg, top1a, top5a, logger = test(model, loader, dev_env, criterion, args, valid_labels, real_labels)

        results = OrderedDict(
            loss=round(loss_avg, 4),
            top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
            top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
            param_count=round(param_count / 1e6, 2),
            img_size=data_config['input_size'][-1],
            cropt_pct=eval_pp_cfg.crop_pct,
            interpolation=data_config['interpolation'])
        logger.log_phase(phase='eval', name_map={'top1': 'Acc@1', 'top5': 'Acc@5'}, **results)

        return results


################
def test_imagenet_c(distortion_name,
                     model,
                     dev_env,
                     criterion,
                     args,
                     transform,
                     severities=list(range(1, 6)),
                     file=sys.stdout):
    errs = []

    for severity in severities:
        valdir = os.path.join(args.data, distortion_name, str(severity))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        _, top1a, _, _ = test(model, val_loader, dev_env, criterion, args)
        errs.append(1. - top1a / 100.)

    print('\n=Average', tuple(errs), file=file)
    return np.mean(errs)
#################


def test(model, loader, dev_env, criterion, args, valid_labels=None, real_labels=None):
    logger = Monitor(logger=_logger)
    tracker = Tracker()
    losses = AvgTensor()
    accuracy = AccuracyTopK(dev_env=dev_env)

    model.eval()
    num_steps = len(loader)
    with torch.no_grad():
        tracker.mark_iter()
        for step_idx, (sample, target) in enumerate(loader):
            last_step = step_idx == num_steps - 1
            tracker.mark_iter_data_end()

            # if sample.device != dev_env.device:
            sample = sample.to(dev_env.device)
            target = target.to(dev_env.device)
            # compute output

            with dev_env.autocast():
                output = model(sample)

            if args.evaluate_imagenet_r:
                output = output[:, imagenet_r_mask]

            if valid_labels is not None:
                output = output[:, valid_labels]

            loss = criterion(output, target)

            if dev_env.type_xla:
                dev_env.mark_step()
            elif dev_env.type_cuda:
                dev_env.synchronize()
            tracker.mark_iter_step_end()

            if real_labels is not None:
                real_labels.add_result(output)
            losses.update(loss.detach(), sample.size(0))
            accuracy.update(output.detach(), target)

            tracker.mark_iter()
            if last_step or step_idx % args.log_freq == 0:
                top1, top5 = accuracy.compute().values()
                loss_avg = losses.compute()
                logger.log_step(
                    phase='eval',
                    step=step_idx,
                    num_steps=num_steps,
                    rate=args.batch_size / tracker.iter_time.avg,
                    loss=loss_avg.item(),
                    top1=top1.item(),
                    top5=top5.item(),
                )

    loss_avg = losses.compute().item()

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = accuracy.compute().values()
        top1a, top5a = top1a.item(), top5a.item()

    return loss_avg, top1a, top5a, logger


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
