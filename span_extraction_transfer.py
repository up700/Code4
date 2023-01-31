# -*- coding:utf-8 -*-
import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl
from torch.nn import MSELoss, KLDivLoss
import torch.nn.functional as F

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertConfig
)

from models.modeling_span import Boundary_Alignment
from utils.data_utils_span import load_and_cache_examples, tag_to_id, get_chunks_bio, get_labels, get_target_preds
from utils.eval_span import evaluate_co, evaluate
from utils.config import config
from utils.loss_utils import get_sp_loss, GCELoss

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "span": (Boundary_Alignment, BertConfig, BertTokenizer)
}

torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, tokenizer, t_total, span_num_labels):
    model_class, config_class, _ = MODEL_CLASSES["span"]

    config_fw = config_class.from_pretrained(
        args.span_model_name_or_path+"-sl",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model_fw = model_class.from_pretrained(
        args.span_model_name_or_path+"-sl",
        config=config_fw,
        span_num_labels=span_num_labels,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model_fw.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_span = [
        {
            "params": [p for n, p in span_model_fw.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in span_model_fw.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_span_fw = AdamW(optimizer_grouped_parameters_span, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_span_fw = get_linear_schedule_with_warmup(
        optimizer_span_fw, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        span_model_fw = torch.nn.DataParallel(span_model_fw)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        span_model_fw = torch.nn.parallel.DistributedDataParallel(
            span_model_fw, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    span_model_fw.zero_grad()

    return span_model_fw, optimizer_span_fw, scheduler_span_fw

def initialize_co(args, tokenizer, t_total, span_num_labels):
    model_class, config_class, _ = MODEL_CLASSES["span"]

    config_fw = config_class.from_pretrained(
        args.span_model_name_or_path+"-sl",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model_fw = model_class.from_pretrained(
        args.span_model_name_or_path+"-sl",
        config=config_fw,
        span_num_labels=span_num_labels,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model_fw.to(args.device)

    config_bw = config_class.from_pretrained(
        args.span_model_name_or_path+"-tl",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model_bw = model_class.from_pretrained(
        args.span_model_name_or_path+"-tl",
        config=config_bw,
        span_num_labels=span_num_labels,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model_bw.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_span = [
        {
            "params": [p for n, p in span_model_fw.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in span_model_fw.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_span_fw = AdamW(optimizer_grouped_parameters_span, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_span_fw = get_linear_schedule_with_warmup(
        optimizer_span_fw, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    optimizer_grouped_parameters_span = [
        {
            "params": [p for n, p in span_model_bw.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in span_model_bw.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_span_bw = AdamW(optimizer_grouped_parameters_span, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_span_bw = get_linear_schedule_with_warmup(
        optimizer_span_bw, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        span_model_fw = torch.nn.DataParallel(span_model_fw)
        span_model_bw = torch.nn.DataParallel(span_model_bw)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        span_model_fw = torch.nn.parallel.DistributedDataParallel(
            span_model_fw, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        span_model_bw = torch.nn.parallel.DistributedDataParallel(
            span_model_bw, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    span_model_fw.zero_grad()
    span_model_bw.zero_grad()

    return span_model_fw, optimizer_span_fw, scheduler_span_fw, span_model_bw, optimizer_span_bw, scheduler_span_bw

def validation_co(args, span_model_fw, span_model_bw, tokenizer, id_to_label_span, pad_token_label_id, best_dev_fw, best_test_fw, best_dev_bw, best_test_bw,\
         global_step, t_total, epoch):
    best_dev_fw, best_dev_bw, is_updated_dev = evaluate_co(args, span_model_fw, span_model_bw, tokenizer, \
        id_to_label_span, pad_token_label_id, best_dev_fw, best_dev_bw, mode="dev", logger=logger, \
        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
    best_test_fw, best_test_bw, is_updated_test = evaluate_co(args, span_model_fw, span_model_bw, tokenizer, \
        id_to_label_span, pad_token_label_id, best_test_fw, best_test_bw, mode="test", logger=logger, \
        prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    if args.local_rank in [-1, 0] and is_updated_dev[0]:
        path = os.path.join(args.output_dir, "checkpoint-best-fw")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            span_model_fw.module if hasattr(span_model_fw, "module") else span_model_fw
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)

    if args.local_rank in [-1, 0] and is_updated_dev[1]:
        path = os.path.join(args.output_dir, "checkpoint-best-bw")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            span_model_bw.module if hasattr(span_model_bw, "module") else span_model_bw
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)

    return best_dev_fw, best_test_fw, best_dev_bw, best_test_bw, is_updated_dev

def validation(args, span_model, tokenizer, id_to_label_span, pad_token_label_id, best_dev, best_test,\
         global_step, t_total, epoch):
    best_dev, is_updated_dev = evaluate(args, span_model, tokenizer, \
        id_to_label_span, pad_token_label_id, best_dev, mode="dev", logger=logger, \
        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
    best_test, is_updated_test = evaluate(args, span_model, tokenizer, \
        id_to_label_span, pad_token_label_id, best_test, mode="test", logger=logger, \
        prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    if args.local_rank in [-1, 0] and is_updated_dev:
        path = os.path.join(args.output_dir, "checkpoint-best")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            span_model.module if hasattr(span_model, "module") else span_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)

    return best_dev, best_test, is_updated_dev


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train_source(args, train_dataset, id_to_label_span, tokenizer, pad_token_label_id):
    """ Train the model """
    span_num_labels = len(id_to_label_span)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    span_model, optimizer_span, scheduler_span = initialize(args, tokenizer, t_total, span_num_labels)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev_bio, best_test_bio = [0, 0, 0], [0, 0, 0]

    loss_funct = MSELoss()

    len_dataloader = len(train_dataloader)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            span_model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "label_ids": batch[2]}
            outputs_span = span_model(**inputs)
            loss1 = outputs_span[0]

            loss = loss1

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                optimizer_span.step()
                scheduler_span.step()  # Update learning rate schedule
                span_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        logger.info("***** training loss : %.4f *****", loss.item())
                        best_dev_bio, best_test_bio, _ = validation(args, span_model, tokenizer, \
                            id_to_label_span, pad_token_label_id, best_dev_bio, best_test_bio, \
                            global_step, t_total, epoch)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev_bio, best_test_bio)

    return results

def co_train(args, train_dataset, id_to_label_span, tokenizer, pad_token_label_id, train_dataset_sl):
    """ Train the model """
    # num_labels = len(labels)
    span_num_labels = len(id_to_label_span)

    path = os.path.join(args.output_dir[:-1], "checkpoint-best")

    sl_model = MODEL_CLASSES["span"][0].from_pretrained(path, span_num_labels=span_num_labels, device=args.device)
    sl_model.to(args.device)
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    train_sampler_sl = RandomSampler(train_dataset_sl) if args.local_rank==-1 else DistributedSampler(train_dataset_sl)
    train_dataloader_sl = DataLoader(train_dataset_sl, sampler=train_sampler_sl, batch_size=args.train_batch_size//2)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    span_model_fw, optimizer_span_fw, scheduler_span_fw, \
        span_model_bw, optimizer_span_bw, scheduler_span_bw = initialize_co(args, tokenizer, t_total, span_num_labels)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    updated_step = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev_fw, best_test_fw = [0, 0, 0], [0, 0, 0]
    best_dev_bw, best_test_bw = [0, 0, 0], [0, 0, 0]

    ce = CrossEntropyLoss().cuda()
    kl_loss = KLDivLoss(reduction="batchmean")
    mse = torch.nn.MSELoss(size_average=True)
    soft_loss = KLDivLoss(reduction='batchmean')
    gce = GCELoss(q=args.q, ignore_index=-100)

    sp_param_fw = nn.Parameter(torch.tensor(args.sp_param_fw).cuda(), requires_grad=True)
    sp_param_bw = nn.Parameter(torch.tensor(args.sp_param_bw).cuda(), requires_grad=True)

    iterator = iter(cycle(train_dataloader_sl))
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            span_model_fw.train()
            span_model_bw.train()
            loss = 0.0
            batch = tuple(t.to(args.device) for t in batch)
            batch_sl = next(iterator)
            batch_sl = tuple(t.to(args.device) for t in batch_sl)
            """
            dataset = TensorDataset(all_input_ids_fw, all_input_ids_bw, \
                all_input_mask_fw, all_input_mask_bw, \
                all_segment_ids_fw, all_segment_ids_bw, \
                all_label_ids_fw, all_label_ids_bw, \
                all_label_mask_fw, all_label_mask_bw, \
                all_order_fw, all_order_bw \
            )

            """

            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            with torch.no_grad():
                outputs_sl = sl_model(**inputs)
                kl_target = F.softmax(outputs_sl[2], dim=-1).detach()

            outputs_fw = span_model_fw(**inputs)
            outputs_bw = span_model_bw(**inputs)

            kl_fw = F.log_softmax(outputs_fw[2], dim=-1)
            kl_bw = F.log_softmax(outputs_bw[2], dim=-1)

            kl_loss_fw = kl_loss(kl_fw, kl_target)
            kl_loss_bw = kl_loss(kl_bw, kl_target)

            cr = mse(outputs_fw[3], outputs_bw[3])

            loss += kl_loss_fw
            loss += kl_loss_bw

            loss += cr

            # Source language
            inputs_sl = {"input_ids": batch_sl[0], "attention_mask": batch_sl[1], "label_ids": batch_sl[2]}
            outputs_fw_sl = span_model_fw(**inputs_sl)
            outputs_bw_sl = span_model_bw(**inputs_sl)
            cr_sl = mse(outputs_fw_sl[3], outputs_bw_sl[3])
            loss_fw_sl = outputs_fw_sl[0]
            loss_bw_sl = outputs_bw_sl[0]

            loss += cr_sl

            loss += loss_fw_sl
            loss += loss_bw_sl

            # Co Supervised
            if epoch > args.cos_start:
                pseudo_fw, top_prob_fw, threshold_fw = get_target_preds(args.threshold, outputs_fw[2])
                pseudo_bw, top_prob_bw, threshold_bw = get_target_preds(args.threshold, outputs_bw[2])

                cos_mask_fw = torch.ge(top_prob_fw, threshold_fw)
                cos_mask_fw = torch.nonzero(cos_mask_fw).squeeze()

                cos_mask_bw = torch.ge(top_prob_bw, threshold_bw)
                cos_mask_bw = torch.nonzero(cos_mask_bw).squeeze()

                if cos_mask_fw.dim() > 0 and cos_mask_bw.dim() > 0:
                    if cos_mask_fw.numel() > 0 and cos_mask_bw.numel() > 0:
                        cos_mask = min(cos_mask_fw.size(0), cos_mask_bw.size(0))
                        cos_fw_loss = ce(outputs_fw[2][cos_mask_bw[:cos_mask]], pseudo_bw[cos_mask_bw[:cos_mask]].cuda().detach())
                        cos_bw_loss = ce(outputs_bw[2][cos_mask_fw[:cos_mask]], pseudo_fw[cos_mask_fw[:cos_mask]].cuda().detach())

                        loss += cos_fw_loss
                        loss += cos_bw_loss

            # Self Penalization
            if epoch <= args.penal_start:
                pseudo_fw, top_prob_fw, threshold_fw = get_target_preds(args.threshold, outputs_fw[2])
                pseudo_bw, top_prob_bw, threshold_bw = get_target_preds(args.threshold, outputs_bw[2])

                sp_mask_fw = torch.lt(top_prob_fw, threshold_fw)
                sp_mask_fw = torch.nonzero(sp_mask_fw).squeeze()

                sp_mask_bw = torch.lt(top_prob_bw, threshold_bw)
                sp_mask_bw = torch.nonzero(sp_mask_bw).squeeze()

                if sp_mask_fw.dim() > 0 and sp_mask_bw.dim() > 0:
                    if sp_mask_fw.numel() > 0 and sp_mask_bw.numel() > 0:
                        sp_mask = min(sp_mask_fw.size(0), sp_mask_bw.size(0))
                        sp_fw_loss = get_sp_loss(outputs_fw[2][sp_mask_fw[:sp_mask]], pseudo_fw[sp_mask_fw[:sp_mask]], sp_param_fw)
                        sp_bw_loss = get_sp_loss(outputs_bw[2][sp_mask_bw[:sp_mask]], pseudo_bw[sp_mask_bw[:sp_mask]], sp_param_bw)

                        loss += sp_fw_loss
                        loss += sp_bw_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                optimizer_span_fw.step()
                optimizer_span_bw.step()
                scheduler_span_fw.step()  # Update learning rate schedule
                scheduler_span_bw.step()
                span_model_fw.zero_grad()
                span_model_bw.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        if epoch <= args.penal_start:
                            try:
                                logger.info("***** training loss : %.4f,  kl_loss_fw: %.4f, kl_loss_bw: %.4f, loss_fw_sl: %.4f, loss_bw_sl: %.4f, sp_fw_loss: %.4f, sp_bw_loss: %.4f, cr: %.4f, cr_sl: %.4f *****", \
                                        loss.item(), kl_loss_fw.item(), kl_loss_bw.item(), loss_fw_sl.item(), loss_bw_sl.item(),
                                        sp_fw_loss.item(), sp_bw_loss.item(), cr.item(), cr_sl.item())
                            except:
                                logger.info("***** training loss : %.4f,  kl_loss_fw: %.4f, kl_loss_bw: %.4f, loss_fw_sl: %.4f, loss_bw_sl: %.4f, sp_fw_loss: %.4f, sp_bw_loss: %.4f, cr: %.4f, cr_sl: %.4f *****", \
                                        loss.item(), kl_loss_fw.item(), kl_loss_bw.item(), loss_fw_sl.item(), loss_bw_sl.item(),
                                        0, 0, cr.item(), cr_sl.item())
                        else:
                            logger.info("***** training loss : %.4f,  kl_loss_fw: %.4f, kl_loss_bw: %.4f, loss_fw_sl: %.4f, loss_bw_sl: %.4f, cos_fw_loss: %.4f, cos_bw_loss: %.4f, cr: %.4f, cr_sl: %.4f *****", \
                                    loss.item(), kl_loss_fw.item(), kl_loss_bw.item(), loss_fw_sl.item(), loss_bw_sl.item(),
                                    cos_fw_loss.item(), cos_bw_loss.item(), cr.item(), cr_sl.item())
                        best_dev_fw, best_test_fw, best_dev_bw, best_test_bw, _ = validation_co(args, span_model_fw, span_model_bw, tokenizer, \
                            id_to_label_span, pad_token_label_id, best_dev_fw, best_test_fw, best_dev_bw, best_test_bw, \
                            global_step, t_total, epoch)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev_fw, best_test_fw, best_dev_bw, best_test_bw)

    return results

def main():
    args = config()
    args.do_train = args.do_train.lower()
    args.do_test = args.do_test.lower()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    id_to_label_span, id_to_label_type = get_labels(args.data_dir, args.dataset)
    # num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = MODEL_CLASSES["span"][2].from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train=="true":

        train_dataset_sl, train_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        logger.info("### Start Training Teacher model on Source language in Span Extraction ###")
        # train_dataset, _ = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        best_results = train_source(args, train_dataset_sl, id_to_label_span, tokenizer, pad_token_label_id)
        logger.info("### Perform Transfer under SSL ###")
        # train_dataset_sl, train_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        best_results = co_train(args, train_dataset, id_to_label_span, tokenizer, pad_token_label_id, train_dataset_sl)

    # Testing
    if args.do_test=="true" and args.local_rank in [-1, 0]:
        predict(args, pad_token_label_id, len(id_to_label_span), mode="dev")

def predict(args, pad_token_label_id, span_num_labels, mode="test"):
    file_path = os.path.join(args.data_dir, "{}_{}.json".format(args.dataset, mode))
    with open(file_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    
    tokenizer = MODEL_CLASSES["span"][2].from_pretrained(args.tokenizer_name_or_path, do_lower_case=args.do_lower_case)
    
    path_fw = os.path.join(args.output_dir, "checkpoint-best-fw")
    span_model_fw = MODEL_CLASSES["span"][0].from_pretrained(path_fw, span_num_labels=span_num_labels, device=args.device)
    span_model_fw.to(args.device)
    
    path_bw = os.path.join(args.output_dir, "checkpoint-best-bw")
    span_model_bw = MODEL_CLASSES["span"][0].from_pretrained(path_bw, span_num_labels=span_num_labels, device=args.device)
    span_model_bw.to(args.device)

    id_to_label_span = {0:"B", 1:"I", 2:"O"}
    
    _, eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)
    span_to_id = {id_to_label_span[id_]:id_ for id_ in id_to_label_span}

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", mode)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_fw = None
    preds_bw = None
    span_label_fw = None
    span_label_bw = None
    out_label_ids = None
    att_mask = None
    span_model_fw.eval()
    span_model_bw.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "label_ids": batch[2], "trainf":False} # 添加segment
            outputs_fw = span_model_fw(**inputs)
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_type": batch[3], "logits_bio": outputs_span[2], "tgt": True}
            outputs_bw = span_model_bw(**inputs)
            fw_logits = outputs_fw[2]
            bw_logits = outputs_bw[2]
            loss1 = outputs_fw[0]
            loss2 = outputs_bw[0]
            loss = loss1+loss2

            if args.n_gpu > 1:
                loss = loss.mean()

            eval_loss += loss.item()

        nb_eval_steps += 1
        
        if preds_fw is None:
            preds_fw = fw_logits.detach() # B, L, C
            preds_bw = bw_logits.detach() # B, L, C
            out_label_ids = batch[2] # B, L
        else:
            preds_fw = torch.cat((preds_fw, fw_logits.detach()), dim=0)
            preds_bw = torch.cat((preds_bw, bw_logits.detach()), dim=0)
            out_label_ids = torch.cat((out_label_ids, batch[2]), dim=0)
    
    eval_loss = eval_loss/nb_eval_steps
    preds_fw = torch.argmax(preds_fw, dim=-1)
    preds_bw = torch.argmax(preds_bw, dim=-1)

    out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list_fw = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list_bw = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_id_list[i].append(out_label_ids[i][j])
                preds_id_list_fw[i].append(preds_fw[i][j])
                preds_id_list_bw[i].append(preds_bw[i][j])

    # correct_preds, total_correct, total_preds = 0., 0., 0. # i variables
    correct_preds_fw, total_correct_fw, total_preds_fw = 0., 0., 0. # i variables
    correct_preds_bw, total_correct_bw, total_preds_bw = 0., 0., 0. # i variables
    new_text = []
    assert len(texts) == len(out_id_list)

    for txt, ground_truth_id, predicted_id_fw, predicted_id_bw in zip(texts, out_id_list, \
                                                            preds_id_list_fw, preds_id_list_bw):
        lab_chunks_bio = get_chunks_bio(ground_truth_id, tag_to_id(args.data_dir, args.dataset))

        txt["esi_gold"] = lab_chunks_bio

        lab_chunks_bio  = set(lab_chunks_bio)

        lab_pred_chunks_fw = get_chunks_bio(predicted_id_fw, tag_to_id(args.data_dir, args.dataset))
        lab_pred_chunks_bw = get_chunks_bio(predicted_id_bw, tag_to_id(args.data_dir, args.dataset))

        txt["esi_pred_fw"] = lab_pred_chunks_fw
        txt["esi_pred_bw"] = lab_pred_chunks_bw

        lab_pred_chunks_fw = set(lab_pred_chunks_fw)
        lab_pred_chunks_bw = set(lab_pred_chunks_bw)

        # Updating the i variables
        correct_preds_fw += len(lab_chunks_bio & lab_pred_chunks_fw)
        total_preds_fw   += len(lab_pred_chunks_fw)
        total_correct_fw += len(lab_chunks_bio)

        correct_preds_bw += len(lab_chunks_bio & lab_pred_chunks_bw)
        total_preds_bw   += len(lab_pred_chunks_bw)
        total_correct_bw += len(lab_chunks_bio)

        new_text.append(txt)
    
    p_fw   = correct_preds_fw / total_preds_fw if correct_preds_fw > 0 else 0
    r_fw   = correct_preds_fw / total_correct_fw if correct_preds_fw > 0 else 0
    new_F_fw  = 2 * p_fw * r_fw / (p_fw + r_fw) if correct_preds_fw > 0 else 0

    p_bw   = correct_preds_bw / total_preds_bw if correct_preds_bw > 0 else 0
    r_bw   = correct_preds_bw / total_correct_bw if correct_preds_bw > 0 else 0
    new_F_bw  = 2 * p_bw * r_bw / (p_bw + r_bw) if correct_preds_bw > 0 else 0

    is_updated = [False, False]

    results = {
       "loss": eval_loss,
       "precision_fw": p_fw,
       "recall_fw": r_fw,
       "f1_fw": new_F_fw,
       "precision_bw": p_bw,
       "recall_bw": r_bw,
       "f1_bw": new_F_bw,
    }

    logger.info("***** Eval results %s *****", mode)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    with open(os.path.join(args.output_dir, args.dataset+"_"+mode+"_pred_spans.json"), "w", encoding="utf-8") as f:
        json.dump(new_text, f, indent=4, ensure_ascii=False)

    return

if __name__ == "__main__":
    main()
