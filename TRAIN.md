## Training Instructions

To train a Robust-ResNet-DW (recommended default), run the following on 8 GPUs each:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py /path/to/ImageNet --val-split val \
--model robust_resnet_dw_small --batch-size 128 --opt adamw --opt-eps 1e-8 --momentum 0.9 --weight-decay 0.05  \
--sched cosine --lr 5e-4 --scale-lr --lr-cycle-decay 1.0 --warmup-lr 1e-6 \
--epochs 300 --decay-epochs 30  --cooldown-epochs 0 \
--aa rand-m9-mstd0.5-inc1 --aug-repeats 3 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
--train-interpolation bicubic --drop-path 0.1 \
--log-interval 200 --checkpoint-interval 10 --checkpoint-hist 1 --workers 6 --pin-mem --seed 0 \
--output /path/to/output_dir --experiment exp_name
```
or the following on TPU V3-8:
```
python3 launch_xla.py --num-devices 8 train.py /path/to/ImageNet --val-split val \
--model robust_resnet_dw_small --batch-size 128 --opt adamw --opt-eps 1e-8 --momentum 0.9 --weight-decay 0.05  \
--sched cosine --lr 5e-4 --scale-lr --lr-cycle-decay 1.0 --warmup-lr 1e-6 \
--epochs 300 --decay-epochs 30  --cooldown-epochs 0 \
--aa rand-m9-mstd0.5-inc1 --aug-repeats 3 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
--train-interpolation bicubic --drop-path 0.1 \
--log-interval 200 --checkpoint-interval 10 --checkpoint-hist 1 --workers 6 --pin-mem --seed 0 \
--output /path/to/output_dir --experiment exp_name
```
- Here the effective batch size is 128 (`batch_size` per gpu) * 8 (gpus per node) = 1024. If memory or # gpus is limited, use `--update_freq` to maintain the effective batch size, which is `batch_size` (per gpu) * 8 (gpus per node) * `update_freq`.
- `lr` is the base learning rate when `--scale-lr` is specified. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 512.
- The exact same hyper-parameters and configs (initialization, augmentation, etc.) are used in GPU and TPU implementations. In practice, we find that GPU training usually leads to slightly better performance than TPU training(~0.2% on ImageNet for 300 epochs). We also observe that shorter training time may lead to a larger gap (~0.5% on ImageNet for 100 epochs). Thus a longer training time is suggested for TPU trainning
- Note that we directly borrow the DeiT training recipe in our paper to eliminate the difference brought by training recipes. 
- Training time is ~42h on TPU V3-8 (300 epochs) or ~50h on 8 A5000 GPU.