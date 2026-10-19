import argparse, os, sys, time, warnings, xlsxwriter
os.environ['NO_ALBUMENTATIONS_UPDATE']="1"
import multiprocessing as mproc
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import *
from model import build_model as change_model

sys.path.append(os.path.abspath("."))
warnings.filterwarnings("ignore")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LEVIR')
    parser.add_argument('--data_dir', type=str, default='data/LEVIR/ABLabel')
    parser.add_argument('--model_size', type=str, default="small")
    parser.add_argument('--pretrain_dir', type=str, default='ckpts/dma_s_in1k_ema.pth')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--last_ckpt_only', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16)  # global batch size of ddp
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=mproc.cpu_count())
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--warmup_multiplier', type=int, default=10)
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+')
    parser.add_argument('--lr_decay_steps', type=int, default=20)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--clip', type=float, default=0.5)
    opt, _ = parser.parse_known_args()
    opt.output_dir = os.path.join(opt.output_dir, opt.dataset, f"ChangeDMA-{opt.model_size[0]}")
    return opt


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def is_main_process():
    return dist.get_rank() == 0


def build_loader(opt, world_size):
    train_data = ChangeDetectionDataset(opt.data_dir, "train")
    val_data = ChangeDetectionDataset(opt.data_dir, "val")
    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    train_bs = max(1, opt.batch_size // world_size)
    val_bs = world_size
    train_loader = DataLoader(train_data,
        batch_size=train_bs,
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=val_bs,
        sampler=val_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, val_loader, train_sampler, val_sampler


def build_model(opt, device):
    model = change_model(img_size=256, num_classes=1, scale=opt.model_size)
    if opt.pretrain_dir is not None:
        pretrained_dict = torch.load(opt.pretrain_dir, map_location="cpu")
        model_dict = model.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            k = "encoder." + k
            if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        if is_main_process():
            print(f"No load key from pretrain: {no_load_key}")
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    model = model.to(device)
    model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    return model


def reduce_mean(tensor):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def gather_for_metric(pred, gt):
    pred_list = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
    gt_list = [torch.zeros_like(gt) for _ in range(dist.get_world_size())]
    dist.all_gather(pred_list, pred)
    dist.all_gather(gt_list, gt)
    pred = torch.cat(pred_list, dim=0)
    gt = torch.cat(gt_list, dim=0)
    return pred, gt


def main(opt):
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if is_main_process():
        os.makedirs(opt.output_dir, exist_ok=True)
    dist.barrier()
    train_loader, val_loader, train_sampler, val_sampler = build_loader(opt, world_size)
    if is_main_process():
        print(f"Length of training dataset: {len(train_loader.dataset)}")
        print(f"Length of val dataset: {len(val_loader.dataset)}")
    model = build_model(opt, device)
    if is_main_process():
        params = sum(p.numel() for p in model.module.parameters())
        print(f"Number of model parameters {params}")
        train_hook = xlsxwriter.Workbook(f'ChangeDMA-{opt.model_size[0]}_{opt.dataset}_train.xlsx')
        train_record = train_hook.add_worksheet()
        val_hook = xlsxwriter.Workbook(f'ChangeDMA-{opt.model_size[0]}_{opt.dataset}_val.xlsx')
        val_record = val_hook.add_worksheet()
        for record in [train_record, val_record]:
            record.write('A1', 'epoch')
            record.write('B1', 'Pre')
            record.write('C1', 'Recall')
            record.write('D1', 'F1')
            record.write('E1', 'IoU')
            record.write('F1', 'acc')
            record.write('G1', 'loss')
        with open(os.path.join(opt.output_dir, 'res.out'), 'w') as f:
            f.write('---------------------------------------------\n')
    else:
        train_record = val_record = train_hook = val_hook = None
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), opt.lr / 10.0 * opt.batch_size,
                                    momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError(f"Unimplemented optimizer type: {opt.optim}")
    scheduler = get_scheduler(optimizer, len(train_loader), opt)
    row, col = 1, 0
    for epoch in range(1, opt.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        tic = time.time()
        if is_main_process():
            print(f"Epoch {epoch}, train:")
        train(train_loader, model, optimizer, BCEDICE_loss, scheduler, epoch, train_record, row, col, opt, device)
        if is_main_process():
            print('Epoch {}, total time {:.2f}, learning_rate {}'.format(
                  epoch, time.time() - tic, optimizer.param_groups[0]['lr']))
            print(f"Epoch {epoch}, val:")
        val(val_loader, model, BCEDICE_loss, epoch, val_record, row, col, opt, device)
        if is_main_process():
            torch.save(model.module.state_dict(), os.path.join(opt.output_dir, "last.pth"))
            print(f"Model saved: {os.path.join(opt.output_dir, 'last.pth')}!")
            if (epoch >= 30) and (not opt.last_ckpt_only):
                torch.save(model.module.state_dict(), os.path.join(opt.output_dir, f"Epoch{epoch}.pth"))
                print(f"Model saved: {os.path.join(opt.output_dir, f'Epoch{epoch}.pth')}!")
            row += 1
        dist.barrier()
    if is_main_process():
        train_hook.close()
        val_hook.close()
    dist.destroy_process_group()


def train(train_loader, model, optimizer, criterion, scheduler, epoch, train_record, row, col, opt, device):
    model.train()
    tool4metric = ConfuseMatrixMeter(n_class=2)
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        imageA, imageB, gts = pack
        imageA = imageA.to(device, non_blocking=True).float()
        imageB = imageB.to(device, non_blocking=True).float()
        gts = gts.to(device, non_blocking=True).float()
        pred_s = model(imageA, imageB)
        if torch.isnan(pred_s).any():
            raise RuntimeError("nan occur")
        gts = torch.unsqueeze(gts, dim=1)
        loss = criterion(pred_s, gts)
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        scheduler.step()
        reduced_loss = reduce_mean(loss.detach())
        loss_record.update(reduced_loss, opt.batch_size)
        bin_pred = (pred_s.detach() > 0.5).int()
        gt_mask = gts.detach().int()
        bin_pred, gt_mask = gather_for_metric(bin_pred, gt_mask)
        if is_main_process():
            tool4metric.update_cm(pr=bin_pred.cpu().numpy(), gt=gt_mask.cpu().numpy())
            if i % 100 == 0 or i == len(train_loader):
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(
                    epoch, opt.epochs, i, len(train_loader), loss_record.show()))
    if is_main_process():
        scores = tool4metric.get_scores()
        print('---------------------------------------------')
        train_record.write(row, col, epoch)
        train_record.write(row, col + 1, scores['precision'])
        train_record.write(row, col + 2, scores['recall'])
        train_record.write(row, col + 3, scores['F1'])
        train_record.write(row, col + 4, scores['iou'])
        train_record.write(row, col + 5, scores['acc'])
        train_record.write(row, col + 6, loss_record.show())


def val(val_loader, model, criterion, epoch, val_record, row, col, opt, device):
    model.eval()
    tool4metric = ConfuseMatrixMeter(n_class=2)
    loss_record = AvgMeter()
    with torch.no_grad():
        for i, pack in enumerate(val_loader, start=1):
            imageA, imageB, gts = pack
            imageA = imageA.to(device, non_blocking=True).float()
            imageB = imageB.to(device, non_blocking=True).float()
            gts = gts.to(device, non_blocking=True).float()
            pred_s = model(imageA, imageB)
            if torch.isnan(pred_s).any():
                pred_s = torch.nan_to_num(pred_s, nan=1.0)
            gts = torch.unsqueeze(gts, dim=1)
            loss = criterion(pred_s, gts)
            reduced_loss = reduce_mean(loss.detach())
            loss_record.update(reduced_loss, opt.batch_size)
            bin_pred = (pred_s.detach() > 0.5).int()
            gt_mask = gts.detach().int()
            bin_pred, gt_mask = gather_for_metric(bin_pred, gt_mask)
            if is_main_process():
                tool4metric.update_cm(pr=bin_pred.cpu().numpy(), gt=gt_mask.cpu().numpy())
                if i % 100 == 0 or i == len(val_loader):
                    print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(
                        epoch, opt.epochs, i, len(val_loader), loss_record.show()))
    if is_main_process():
        scores = tool4metric.get_scores()
        print(f"IoU for epoch {epoch} is {scores['iou']}")
        print(f"F1 for epoch {epoch} is {scores['F1']}")
        print(f"acc for epoch {epoch} is {scores['acc']}")
        print(f"precision for epoch {epoch} is {scores['precision']}")
        print(f"recall for epoch {epoch} is {scores['recall']}")
        print('---------------------------------------------')
        with open(os.path.join(opt.output_dir, 'res.out'), 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"IoU for epoch {epoch}: {scores['iou']}\n")
            f.write(f"F1 for epoch {epoch}: {scores['F1']}\n")
            f.write(f"acc for epoch {epoch}: {scores['acc']}\n")
            f.write(f"precision for epoch {epoch}: {scores['precision']}\n")
            f.write(f"recall for epoch {epoch}: {scores['recall']}\n")
            f.write(f"Loss for epoch {epoch}: {loss_record.show()}\n")
            f.write('---------------------------------------------\n')
        val_record.write(row, col, epoch)
        val_record.write(row, col + 1, scores['precision'])
        val_record.write(row, col + 2, scores['recall'])
        val_record.write(row, col + 3, scores['F1'])
        val_record.write(row, col + 4, scores['iou'])
        val_record.write(row, col + 5, scores['acc'])
        val_record.write(row, col + 6, loss_record.show())


if __name__ == '__main__':
    opt = parse_option()
    main(opt)
