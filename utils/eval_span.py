# -*- coding:utf-8 -*-
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.data_utils_span import load_and_cache_examples, tag_to_id, get_chunks_bio
from flashtool import Logger

def evaluate_co(args, span_model_fw, span_model_bw, tokenizer, id_to_label_span, \
    pad_token_label_id, best_fw, best_bw, mode, logger, prefix="", verbose=True):
    
    _, eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)
    span_to_id = {id_to_label_span[id_]:id_ for id_ in id_to_label_span}
    non_entity_id = span_to_id["O"]
    num_class = len(span_to_id)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
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
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "label_ids": batch[2], "trainf":False}
            outputs_fw = span_model_fw(**inputs)
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

    # preds: nb, type_num, L, span_num
    # out_label_ids: nb*type_num, L
    
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
    # print("EVAL:")

    for ground_truth_id, predicted_id_fw, predicted_id_bw in zip(out_id_list, \
                                                            preds_id_list_fw, preds_id_list_bw):
        lab_chunks_bio = get_chunks_bio(ground_truth_id, tag_to_id(args.data_dir, args.dataset))

        lab_chunks_bio  = set(lab_chunks_bio)

        lab_pred_chunks_fw = get_chunks_bio(predicted_id_fw, tag_to_id(args.data_dir, args.dataset))
        lab_pred_chunks_bw = get_chunks_bio(predicted_id_bw, tag_to_id(args.data_dir, args.dataset))

        lab_pred_chunks_fw = set(lab_pred_chunks_fw)
        lab_pred_chunks_bw = set(lab_pred_chunks_bw)

        # Updating the i variables
        correct_preds_fw += len(lab_chunks_bio & lab_pred_chunks_fw)
        total_preds_fw   += len(lab_pred_chunks_fw)
        total_correct_fw += len(lab_chunks_bio)

        correct_preds_bw += len(lab_chunks_bio & lab_pred_chunks_bw)
        total_preds_bw   += len(lab_pred_chunks_bw)
        total_correct_bw += len(lab_chunks_bio)

    p_fw   = correct_preds_fw / total_preds_fw if correct_preds_fw > 0 else 0
    r_fw   = correct_preds_fw / total_correct_fw if correct_preds_fw > 0 else 0
    new_F_fw  = 2 * p_fw * r_fw / (p_fw + r_fw) if correct_preds_fw > 0 else 0

    p_bw   = correct_preds_bw / total_preds_bw if correct_preds_bw > 0 else 0
    r_bw   = correct_preds_bw / total_correct_bw if correct_preds_bw > 0 else 0
    new_F_bw  = 2 * p_bw * r_bw / (p_bw + r_bw) if correct_preds_bw > 0 else 0

    is_updated = [False, False]

    if new_F_fw > best_fw[-1]:
        best_fw = [p_fw, r_fw, new_F_fw]
        is_updated[0] = True

    if new_F_bw > best_bw[-1]:
        best_bw = [p_bw, r_bw, new_F_bw]
        is_updated[1] = True

    results = {
       "loss": eval_loss,
       "precision_fw": p_fw,
       "recall_fw": r_fw,
       "f1_fw": new_F_fw,
       "best_precision_fw": best_fw[0],
       "best_recall_fw": best_fw[1],
       "best_f1_fw": best_fw[-1],
       "precision_bw": p_bw,
       "recall_bw": r_bw,
       "f1_bw": new_F_bw,
       "best_precision_bw": best_bw[0],
       "best_recall_bw": best_bw[1],
       "best_f1_bw": best_bw[-1]
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return best_fw, best_bw, is_updated

def evaluate(args, span_model, tokenizer, id_to_label_span, pad_token_label_id, best_bio, mode, logger, prefix="", verbose=True):
    
    _, eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)
    span_to_id = {id_to_label_span[id_]:id_ for id_ in id_to_label_span}
    non_entity_id = span_to_id["O"]
    num_class = len(span_to_id)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    span_label = None
    out_label_ids = None
    att_mask = None
    span_model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "label_ids": batch[2]}
            outputs = span_model(**inputs)
            logits = outputs[2]
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()

            eval_loss += loss.item()

        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach() # B, L, C
            out_label_ids = batch[2] # B, L
        else:
            preds = torch.cat((preds, logits.detach()), dim=0)
            out_label_ids = torch.cat((out_label_ids, batch[2]), dim=0)

    # preds: nb, type_num, L, span_num
    # out_label_ids: nb*type_num, L
    
    eval_loss = eval_loss/nb_eval_steps

    preds = torch.argmax(preds, dim=-1)

    out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_id_list[i].append(out_label_ids[i][j])
                preds_id_list[i].append(preds[i][j])

    # correct_preds, total_correct, total_preds = 0., 0., 0. # i variables
    correct_preds, total_correct, total_preds = 0., 0., 0. # i variables
    # print("EVAL:")

    for ground_truth_id, predicted_id in zip(out_id_list, preds_id_list):
        lab_chunks_bio = get_chunks_bio(ground_truth_id, tag_to_id(args.data_dir, args.dataset))

        lab_chunks_bio  = set(lab_chunks_bio)

        lab_pred_chunks = get_chunks_bio(predicted_id, tag_to_id(args.data_dir, args.dataset))

        lab_pred_chunks = set(lab_pred_chunks)

        # Updating the i variables
        correct_preds += len(lab_chunks_bio & lab_pred_chunks)
        total_preds   += len(lab_pred_chunks)
        total_correct += len(lab_chunks_bio)


    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    is_updated = False

    if new_F > best_bio[-1]:
        best_bio = [p, r, new_F]
        is_updated = True

    results = {
       "loss": eval_loss,
       "precision": p,
       "recall": r,
       "f1": new_F,
       "best_precision": best_bio[0],
       "best_recall": best_bio[1],
       "best_f1": best_bio[-1]
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return best_bio, is_updated
