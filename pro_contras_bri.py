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
from torch.utils.data import DataLoader, RandomSampler
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
from utils.data_utils_pcb import load_and_cache_examples, get_labels, get_target_preds, soft_label
from utils.config import config

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "pre_finetune": (Boundary_Alignment, BertConfig, BertTokenizer),
}

torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, tokenizer, t_total, span_num_labels):
    model_class, config_class, _ = MODEL_CLASSES["pre_finetune"]

    config_fw = config_class.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_fw = model_class.from_pretrained(
        args.model_name_or_path,
        config=config_fw,
        span_num_labels=span_num_labels,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_fw.to(args.device)

    # t = sum([param.nelement() for param in span_model_fw.parameters()])
    # print(t/1e6)
    # exit()

    config_bw = config_class.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_bw = model_class.from_pretrained(
        args.model_name_or_path,
        config=config_bw,
        span_num_labels=span_num_labels,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_bw.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_fw.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_fw.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_fw = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_fw = get_linear_schedule_with_warmup(
        optimizer_fw, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_bw.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_bw.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_bw = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_bw = get_linear_schedule_with_warmup(
        optimizer_bw, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_fw = torch.nn.DataParallel(model_fw)
        model_bw = torch.nn.DataParallel(model_bw)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_fw = torch.nn.parallel.DistributedDataParallel(
            model_fw, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        model_bw = torch.nn.parallel.DistributedDataParallel(
            model_bw, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model_fw.zero_grad()
    model_bw.zero_grad()

    return model_fw, optimizer_fw, scheduler_fw, model_bw, optimizer_bw, scheduler_bw

def contrastive_loss(h_fw, h_bw, order_fw, order_bw):
    # h_fw: N, L_f, d
    # h_bw: N, L_b, d
    # order_fw/order_bw: N, L
    N_f, L_f = order_fw.size()
    N_b, L_b = order_bw.size()
    assert N_f==N_b
    order_fw_expd = order_fw.unsqueeze(2).expand(N_f, L_f, L_b) # N_f, L_f, L_b
    order_bw_expd = order_bw.unsqueeze(1).expand(N_f, L_f, L_b)
    order_mask = (order_fw_expd==order_bw_expd)&(order_fw_expd>0)&(order_bw_expd>0) # N_f, L_f, L_b
    s = torch.bmm(h_fw, h_bw.permute(0,2,1)) # N, L_f, L_b
    # # h_fw_expd = h_fw.unsqueeze(2).expand()
    # f = torch.norm(h_fw, None, 2).unsqueeze(2).expand(N_f, L_f, L_b) # N, L_f, L_b
    # b = torch.norm(h_bw, None, 2).unsqueeze(1).expand(N_f, L_f, L_b) # N, L_f, L_b
    # s = s/(f*b) # N, L_f, L_b
    s_norm = F.log_softmax(s/0.1, dim=-1) # N, L_f, L_b
    loss_cs = torch.mean(-s_norm.view(-1)[order_mask.view(-1)])
    # loss_funct = NLLLoss()
    # loss_funct()
    # output = F.cosine_similarity(input1, input2, dim=0)
    return loss_cs

def train(args, train_dataset, id_to_label_span, tokenizer, pad_token_label_id):
    """ Train the model """
    # num_labels = len(labels)
    span_num_labels = len(id_to_label_span)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    model_fw, optimizer_fw, scheduler_fw, \
        model_bw, optimizer_bw, scheduler_bw = initialize(args, tokenizer, t_total, span_num_labels)

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
    best_dev_fw, best_test_fw = [0, 0, 0], [0, 0, 0]
    best_dev_bw, best_test_bw = [0, 0, 0], [0, 0, 0]

    len_dataloader = len(train_dataloader)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model_fw.train()
            model_bw.train()
            batch = tuple(t.to(args.device) for t in batch)
            """
            dataset = TensorDataset(all_input_ids_sl, all_input_ids_tl, \
                all_input_mask_sl, all_input_mask_tl, \
                all_label_ids_sl, all_label_ids_tl, \
                all_label_mask_sl, all_label_mask_tl, \
                all_order_sl, all_order_tl \
            )

            """

            inputs_sl = {"input_ids": batch[0], "attention_mask": batch[2], \
                            "label_ids": batch[4], "label_mask": batch[6], "order": batch[8]}
            outputs_sl = model_fw(**inputs_sl)

            inputs_tl = {"input_ids": batch[1], "attention_mask": batch[3], \
                            "label_ids": batch[5], "label_mask": batch[7], "order": batch[9]}
            outputs_tl = model_bw(**inputs_tl)

            loss = contrastive_loss(outputs_sl[1], outputs_tl[1], batch[8], batch[9])

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                optimizer_fw.step()
                optimizer_bw.step()
                scheduler_fw.step()  # Update learning rate schedule
                scheduler_bw.step()
                model_fw.zero_grad()
                model_bw.zero_grad()
                global_step += 1
                logger.info("***** training loss : %.4f *****", loss.item())

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    save_model(args, model_fw, tokenizer, flag="sl")
    save_model(args, model_bw, tokenizer, flag="tl")

    # results = (best_dev_fw, best_test_fw, best_dev_bw, best_test_bw)

    # return results

def save_model(args, model, tokenizer, flag="sl"):
    path = os.path.join(args.output_dir, "checkpoint-best-bpt-"+flag)
    logger.info("Saving model checkpoint to %s", path)
    if not os.path.exists(path):
        os.makedirs(path)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(path)
    tokenizer.save_pretrained(path)

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

    tokenizer = MODEL_CLASSES["pre_finetune"][2].from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train=="true":
        train_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        best_results = train(args, train_dataset,\
            id_to_label_span, tokenizer, pad_token_label_id)


if __name__ == "__main__":
    main()
