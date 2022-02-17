# Self-Supervised Vision Transformers for Cloud segmentation and characterization Cloud-DINO

PyTorch implementation for Cloud-DINO.  

## For training Cloud-DINO

`python3 -m torch.distributed.launch --nproc_per_node=1 cloud_dino_training.py --data_path /path/to/your/sky_images/ --output_dir /path/to/your/model/ --use_fp16 false`

## Running inference to obtain Cloud-DINO's features

`python3 -m torch.distributed.launch --nproc_per_node=1 cloud_dino_inference.py --data_path /path/to/your/sky_images/ --pretrained_weights /path/to/your/model/checkpoint0000.pth --dump_features /path/to/your/features/`
