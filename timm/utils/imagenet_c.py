imagenetc_distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

imagenetc_alexnet_error_rates_list = [
    (0.69528, 0.82542, 0.93554, 0.98138, 0.99452), # gaussian_noise
    (0.71224, 0.85108, 0.93574, 0.98182, 0.99146), # shot_noise
    (0.78374, 0.89808, 0.9487, 0.9872, 0.99548), # impulse_noise
    (0.656239999999999, 0.73202, 0.85036, 0.91364, 0.94714), # defocus_blur
    (0.64308, 0.75054, 0.88806, 0.91622, 0.93344), # glass_blur
    (0.5843, 0.70048, 0.82108, 0.8975, 0.92638), # motion_blur
    (0.70008, 0.769919999999999, 0.80784, 0.84198, 0.87198), # zoom_blur
    (0.71726, 0.88392, 0.86468, 0.9187, 0.94952), # snow
    (0.6139, 0.797339999999999, 0.8879, 0.89942, 0.9343), # frost
    (0.67474, 0.7605, 0.84378, 0.8726, 0.945), # fog
    (0.4514, 0.48502, 0.54048, 0.62166, 0.724399999999999), # brightness
    (0.64548, 0.7615, 0.88874, 0.9776, 0.9927), # contrast
    (0.52596, 0.70116, 0.55686, 0.64076, 0.80554), # elastic_transform
    (0.52218, 0.5462, 0.737279999999999, 0.87092, 0.91262), # pixelate
    (0.510019999999999, 0.54718, 0.57294, 0.654579999999999, 0.74778), # jpeg_compression
]