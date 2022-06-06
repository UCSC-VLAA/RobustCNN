## Testing Instructions

To test a model on ImageNet, run the following:
```
python validate.py /path/to/ImageNet --model robust_resnet_dw_small --checkpoint /path/to/ckpt \
--batch-size 32 --pin-mem --log-freq 50 --resize_mode resize_center_crop --workers 8
```

To test a model on Stylized-ImageNet, run the following:
```
python validate.py /path/to/Stylized-ImageNet --model robust_resnet_dw_small --checkpoint /path/to/ckpt \
--batch-size 32 --pin-mem --log-freq 50 --resize_mode none  --workers 8
```

To test a model on ImageNet-C, run the following:
```
python validate.py /path/to/ImageNet-C --model robust_resnet_dw_small --checkpoint /path/to/ckpt \
--batch-size 32 --pin-mem --log-freq 50 --resize_mode none --workers 8  --evaluate_imagenet_c --normalize
```
It is worth noting that for ImageNet-C evaluation, the error rate is calculated based on the Noise, Blur, Weather and Digital categories.

To test a model on ImageNet-R, run the following:
```
python validate.py /path/to/ImageNet-R --model robust_resnet_dw_small --checkpoint /path/to/ckpt \
--batch-size 32 --pin-mem --log-freq 50 --evaluate_imagenet_r --resize_mode resize_center_crop --workers 8
```

To test a model on ImageNet-Sketch, run the following:
```
python validate.py /path/to/ImageNet-Sketch --model robust_resnet_dw_small --checkpoint /path/to/ckpt \
--batch-size 32 --pin-mem --log-freq 50 --resize_mode resize_center_crop --workers 8
```