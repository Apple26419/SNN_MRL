import copy
import time
import torch
import random
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.clock_driven import functional
from utils import Bar,  AverageMeter, accuracy
import torch.nn as nn
from modules.margin_resculpted_surrogate import MarginResculptedSurrogate, mrl_loss, mrl_reset, mrl_force_clear



def tra(model, dataset, data, time_step, epoch, optimizer, lr_scheduler, scaler, loss_lambda=0.0, attacker=None, writer=None, gamma=0.0):

    start_time = time.time()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Training', max=len(data))

    model.train()

    train_loss = 0
    train_acc = 0
    train_samples = 0
    batch_idx = 0

    for frame, label in data:
        batch_idx += 1
        frame = frame.float().cuda()
        label = label.cuda()
        label_real = torch.cat([label for _ in range(time_step)], 0)


        if gamma > 0.0:
            mrl_reset(model)

        optimizer.zero_grad()
        out_all = []

        input_frame = frame

        if attacker is not None:
            input_frame = attacker(input_frame, label)

        for t in range(time_step):
            with amp.autocast():
                out = model(input_frame)
                if t == 0:
                    total_frame = out.clone().detach()
                else:
                    total_frame += out.clone().detach()
                out_all.append(out)
        out_all = torch.cat(out_all, 0)

        with amp.autocast():
            if gamma > 0.0:
                ce_loss = F.cross_entropy(out_all, label_real)
                aux_loss = mrl_loss(model)
                p_ref = None
                for p in model.parameters():
                    if p.requires_grad:
                        p_ref = p
                        break
                assert p_ref is not None
                g_ce = torch.autograd.grad(ce_loss, p_ref, retain_graph=True, create_graph=False, allow_unused=True)[0]
                g_aux = torch.autograd.grad(aux_loss, p_ref, retain_graph=True, create_graph=False, allow_unused=True)[0]
                g_ce_norm = g_ce.detach().abs().mean() if g_ce is not None else torch.zeros((), device=ce_loss.device)
                g_aux_norm = g_aux.detach().abs().mean() if g_aux is not None else torch.zeros((), device=ce_loss.device)
                lambda_mrl = (g_ce_norm / (g_aux_norm + 1e-12)).detach()
                loss = ce_loss + lambda_mrl * aux_loss

            else:
                loss = F.cross_entropy(out_all, label_real)

            scaler.scale(loss).backward()



        scaler.step(optimizer)
        scaler.update()

        if gamma > 0.0:
            mrl_force_clear()

        batch_loss = loss.item()
        train_loss += loss.item() * label.numel()

        prec1, prec5 = accuracy(total_frame.data, label.data, topk=(1, 5))
        losses.update(batch_loss, input_frame.size(0))
        top1.update(prec1.item(), input_frame.size(0))
        top5.update(prec5.item(), input_frame.size(0))

        train_samples += label.numel()
        train_acc += (total_frame.argmax(1) == label).float().sum().item()

        functional.reset_net(model)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
            batch=batch_idx,
            size=len(data),
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
        )
        bar.next()

    bar.finish()

    train_loss /= train_samples
    train_acc /= train_samples
    if writer is not None:
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
    lr_scheduler.step()



    return train_loss, train_acc

def val(model, dataset, data, time_step, epoch, optimizer, lr_scheduler, scaler, loss_lambda=0.0, attacker=None, writer=None):
    
    start_time = time.time()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Validating', max=len(data))

    model.eval()

    val_loss = 0
    val_acc = 0
    val_samples = 0
    batch_idx = 0

    for frame, label in data:
        batch_idx += 1
        frame = frame.float().cuda()
        label = label.cuda()
        label_real = torch.cat([label for _ in range(time_step)], 0)

        input_frame = frame
        if attacker is not None:
            input_frame = attacker(input_frame, label)
        out_all = []
        for t in range(time_step):
            with amp.autocast():
                out = model(input_frame)
                if t == 0:
                    total_frame = out.clone().detach()
                else:
                    total_frame += out.clone().detach()
                out_all.append(out)
        out_all = torch.cat(out_all, 0)
        with torch.no_grad():
            if loss_lambda > 0.0:
                label_one_hot = torch.zeros_like(out_all).fill_(1.0).to(out_all.device)
                mse_loss = F.mse_loss(out_all, label_one_hot)
                loss = (1 - loss_lambda) * F.cross_entropy(out_all, label_real) + loss_lambda * mse_loss
            else:
                loss = F.cross_entropy(out_all, label_real)


        batch_loss = loss.item()
        val_loss += loss.item() * label.numel()

        prec1, prec5 = accuracy(total_frame.data, label.data, topk=(1, 5))
        losses.update(batch_loss, input_frame.size(0))
        top1.update(prec1.item(), input_frame.size(0))
        top5.update(prec5.item(), input_frame.size(0))

        val_samples += label.numel()
        val_acc += (total_frame.argmax(1) == label).float().sum().item()

        functional.reset_net(model)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
            batch=batch_idx,
            size=len(data),
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
        )
        bar.next()

    bar.finish()
    val_loss /= val_samples
    val_acc /= val_samples
    if writer is not None:
        writer.add_scalar('train_loss', val_loss, epoch)
        writer.add_scalar('train_acc', val_acc, epoch)
    return val_loss, val_acc


