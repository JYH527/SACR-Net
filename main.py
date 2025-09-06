import argparse
import time
import datetime
import random
from pathlib import Path
import json

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, train_one_epoch_classifier, evaluate_hoi
from models import build_model
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # 学习率和优化器参数
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Base learning rate for transformer parameters')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='Learning rate for backbone parameters')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size, increased for better gradient estimation')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay for regularization')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Number of training epochs, increased for better convergence')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='Number of training epochs, increased for better convergence')
    parser.add_argument('--lr_drop', default=[40,60], type=list,
                        help='Epoch at which to drop learning rate')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Gradient clipping max norm')

    # 模型参数
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='Path to the pretrained model. If set, only the mask head will be trained')
    # 骨干网络参数
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='Name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true',
                        help='If true, replace stride with dilation in the last convolutional block (DC5)')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help='Type of positional embedding to use on top of the image features')
    parser.add_argument('--return_interm_layers', default=[1, 2, 3], type=list,
                        help='Layers of backbone to use')
    parser.add_argument('--backbone_pretrained', default='https://download.pytorch.org/models/resnet50-0676ba61.pth', type=str,
                        help='Path or URL to pretrained backbone weights')

    # Transformer 参数
    parser.add_argument('--enc_layers', default=6, type=int,
                        help='Number of encoding layers in the transformer')
    parser.add_argument('--dec_layers', default=6, type=int,
                        help='Number of decoding layers in the transformer')
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help='Intermediate size of the feedforward layers in the transformer blocks')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout applied in the transformer')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads inside the transformer attentions')
    parser.add_argument('--pre_norm', action='store_true',
                        help='Use pre-normalization in transformer')
    parser.add_argument('--num_levels', default=3,
                        help='Use deformable dec layers')

    # HOI 任务参数
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help='Number of object classes, set for COCO-like datasets')
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help='Number of verb classes, set for HICO-DET')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int,
                        help='Category ID for human/subject in HOI')
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for verb classification, focal loss for imbalanced classes')
    parser.add_argument('--two_stage', action='store_true',
                        help='Use pretrained pos_embed or bbox_embed to attain reference_points')
    parser.add_argument('--num_queries', default=100, type=int,
                        help='Use num_queries to attain qurey_embed')
    parser.add_argument('--return_intermedia', action='store_true',
                        help='Return intermediate outputs from decoder layers')
    parser.add_argument('--fusion_query', default=True, type=bool,
                        help='if fuse query with memory')

    # 损失函数参数
    # * Matcher
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help='Disables auxiliary decoding losses (loss at each layer)')
    parser.add_argument('--use_matching', action='store_true',
                        help='Use obj/sub matching 2-class loss in first decoder')
    parser.add_argument('--set_cost_class', default=1.0, type=float,
                        help='Class coefficient in the matching cost')
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help='L1 box coefficient in the matching cost, increased for precise localization')
    parser.add_argument('--set_cost_giou', default=1.0, type=float,
                        help='GIOU box coefficient in the matching cost, increased for better overlap')
    parser.add_argument('--set_cost_obj_class', default=1.0, type=float,
                        help='Object class coefficient in the matching cost')
    parser.add_argument('--set_cost_verb_class', default=2.0, type=float,
                        help='Verb class coefficient in the matching cost, increased for verb importance')
    parser.add_argument('--set_cost_matching', default=1.0, type=float,
                        help='Sub and obj box matching coefficient in the matching cost')
    parser.add_argument('--mask_loss_coef', default=1.0, type=float,
                        help='Mask loss coefficient')
    parser.add_argument('--dice_loss_coef', default=1.0, type=float,
                        help='Dice loss coefficient')
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float,
                        help='L1 box loss coefficient, increased for precise localization')
    parser.add_argument('--giou_loss_coef', default=2.0, type=float,
                        help='GIou loss coefficient, increased for better overlap')
    parser.add_argument('--obj_loss_coef', default=1.5, type=float,
                        help='Object classification loss coefficient')
    parser.add_argument('--verb_loss_coef', default=1.0, type=float,
                        help='Verb classification loss coefficient, increased for verb importance')
    parser.add_argument('--verb_cardinality_loss_coef', default=0.5, type=float,
                        help='Verb classification loss coefficient, increased for verb importance')
    parser.add_argument('--obj_cardinality_loss_coef', default=2.0, type=float,
                        help='Verb classification loss coefficient, increased for verb importance')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Focal loss alpha, adjusted for class imbalance')
    parser.add_argument('--matching_loss_coef', default=1.0, type=float,
                        help='Matching loss coefficient')
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help='Relative classification weight of the no-object class')

    # 数据集参数
    parser.add_argument('--dataset_file', default='hico', type=str,
                        help='Dataset to use, set to HICO-DET')
    parser.add_argument('--hoi_path', type=str, default='./data/hico_20160224_det',
                        help='Path to HICO-DET dataset')

    # 训练输出和设备
    parser.add_argument('--output_dir', default='./output/vcoco/',
                        help='Path where to save checkpoints and logs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers, increased for faster data loading')

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='URL used to set up distributed training')

    # HOI 评估参数
    parser.add_argument('--use_nms_filter', action='store_true',
                        help='Use pair NMS filter during evaluation')
    parser.add_argument('--thres_nms', default=0.5, type=float,
                        help='NMS threshold for filtering')
    parser.add_argument('--nms_alpha', default=0.7, type=float,
                        help='NMS alpha parameter')
    parser.add_argument('--nms_beta', default=0.7, type=float,
                        help='NMS beta parameter')
    parser.add_argument('--json_file', default='results.json', type=str,
                        help='Output file for evaluation results')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Learning rate decay factor')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze backbone during training')
    parser.add_argument('--ko', action='store_true',
                        help='KO eval mode for HICO-DET')

    return parser

def main(args):
    utils.init_distributed_mode(args)

    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    ]

    if not args.freeze:
        param_dicts.append({
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        })

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop, gamma=args.gamma)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        # Filter weights to load only backbone and encoder parts
        pretrained_dict = checkpoint['model']
        model_dict = model_without_ddp.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if
                             k.startswith(('backbone.', 'transformer.encoder.')) and k in model_dict}
        model_dict.update(filtered_dict)
        model_without_ddp.load_state_dict(model_dict, strict=False)
        print(f"Loaded pretrained weights for backbone and encoder from {args.pretrained}")

    # if args.eval:
    #     test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args)
    #     return

    print("Start training")
    start_time = time.time()
    best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        # if epoch == args.epochs - 1:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_last.pth')
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

        if epoch < args.lr_drop[0] and epoch % 5 != 0:  ## eval every 5 epoch before lr_drop
            continue
        elif epoch >= args.lr_drop[0] and epoch % 2 == 0:  ## eval every 2 epoch after lr_drop
            continue

        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args)
        coco_evaluator = None
        if args.dataset_file == 'hico':
            performance = test_stats['mAP']
        elif args.dataset_file == 'vcoco':
            performance = test_stats['mAP_all']

        if performance > best_performance:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        # if utils.is_main_process() and not args.debug:
        #     wandb.log(log_stats)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

