import argparse
import os
import torch
from tqdm import tqdm
import pdb

from utils import setup_seed
from dataset import Kitti, get_dataloader
from model import PointPillars
from loss import Loss
from torch.utils.tensorboard import SummaryWriter


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


def execute_process(mode, pointpillars, optimizer, scheduler, val_dataloader, dataloder_type, loss_func, epoch,
                    epoch_loss_writer):
    if mode == "eval":
        pointpillars.eval()
    elif mode == "train":
        pointpillars.train()
    val_step = 0
    # with torch.no_grad():
    for i, data_dict in enumerate(tqdm(val_dataloader)):
        if not args.no_cuda:
            # move the tensors to the cuda
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

        if mode == "train":
            optimizer.zero_grad()

        batched_pts = data_dict['batched_pts']
        batched_gt_bboxes = data_dict['batched_gt_bboxes']
        batched_labels = data_dict['batched_labels']
        batched_difficulty = data_dict['batched_difficulty']
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
            pointpillars(batched_pts=batched_pts,
                         mode='train',
                         batched_gt_bboxes=batched_gt_bboxes,
                         batched_gt_labels=batched_labels,
                         drop_out=mode == "train"
                         )

        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

        batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
        batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
        batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
        # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
        batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
        # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)

        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]
        # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
        # bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
        # batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
        batched_dir_labels = batched_dir_labels[pos_idx]

        num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

        loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                              bbox_pred=bbox_pred,
                              bbox_dir_cls_pred=bbox_dir_cls_pred,
                              batched_labels=batched_bbox_labels,
                              num_cls_pos=num_cls_pos,
                              batched_bbox_reg=batched_bbox_reg,
                              batched_dir_labels=batched_dir_labels)
        if mode == "train":
            loss = loss_dict['total_loss']
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

        global_step = epoch * len(val_dataloader) + val_step + 1
        if global_step % args.log_freq == 0:
            save_summary(epoch_loss_writer, loss_dict, global_step, dataloder_type)
        val_step += 1


def main(args):
    setup_seed()
    train_dataset = Kitti(data_root=args.data_root, split='train')
    val_dataset = Kitti(data_root=args.data_root, split='val')
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True)

    val_dataloader = get_dataloader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False)
    load_model_num = int(args.load_model_num)
    dropout_rate = float(args.dropout_rate)
    dir_name = f"checkpoints_{dropout_rate}"
    load_model_path = os.path.join("pillar_logs", dir_name, f"epoch_{load_model_num}.pth")
    print(load_model_path)
    start_epoch = -1
    if os.path.exists(load_model_path):
        if not args.no_cuda:
            pointpillars = PointPillars(nclasses=args.nclasses, dropout_p=dropout_rate).cuda()
            checkpoint = torch.load(load_model_path)

        else:
            pointpillars = PointPillars(nclasses=args.nclasses, dropout_p=dropout_rate)
            checkpoint = torch.load(load_model_path, map_location=torch.device('cpu'))
        print(checkpoint.keys())
        print(checkpoint["epoch"])

        pointpillars.load_state_dict(checkpoint['net'])
        init_lr = args.init_lr
        optimizer = torch.optim.AdamW(params=pointpillars.parameters(),
                                      lr=init_lr,
                                      betas=(0.95, 0.99),
                                      weight_decay=0.01)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(start_epoch)
    else:
        print(f"there is no path named {load_model_path}")
        if not args.no_cuda:
            pointpillars = PointPillars(nclasses=args.nclasses, dropout_p=dropout_rate).cuda()
        else:
            pointpillars = PointPillars(nclasses=args.nclasses, dropout_p=dropout_rate)
        init_lr = args.init_lr
        optimizer = torch.optim.AdamW(params=pointpillars.parameters(),
                                      lr=init_lr,
                                      betas=(0.95, 0.99),
                                      weight_decay=0.01)

    loss_func = Loss()
    max_iters = len(train_dataloader) * args.max_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=init_lr * 10,
                                                    total_steps=max_iters,
                                                    pct_start=0.4,
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True,
                                                    base_momentum=0.95 * 0.895,
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, dir_name)
    os.makedirs(saved_ckpt_path, exist_ok=True)
    saved_epoch_loss_path = os.path.join(args.saved_path, f'{dir_name}/loss')
    os.makedirs(saved_epoch_loss_path, exist_ok=True)
    epoch_loss_writer = SummaryWriter(saved_epoch_loss_path)

    for epoch in range(start_epoch + 1, args.max_epoch):
        print('=' * 20, epoch, "/", args.max_epoch, '=' * 20)
        train_step = 0
        loss_file = []
        execute_process("train", pointpillars, optimizer, scheduler, train_dataloader, "train_dataset", loss_func,
                        epoch,
                        epoch_loss_writer)

        # if (epoch + 1) % args.ckpt_freq_epoch == 0:
        checkpoint = {
            "net": pointpillars.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }

        execute_process("eval", pointpillars, optimizer, scheduler, train_dataloader, "train_dataset", loss_func,
                        epoch, epoch_loss_writer)

        execute_process("eval", pointpillars, optimizer, scheduler, val_dataloader, "val_dataset", loss_func, epoch,
                        epoch_loss_writer)
        torch.save(checkpoint, os.path.join(saved_ckpt_path, f'epoch_{epoch}.pth'))
        if os.path.exists(os.path.join(saved_ckpt_path, f'epoch_{epoch - 1}.pth')):
            os.remove(os.path.join(saved_ckpt_path, f'epoch_{epoch - 1}.pth'))
        pointpillars.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti',
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--load_model_num', default=0)
    parser.add_argument('--dropout_rate', default=0)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
