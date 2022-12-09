#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import socket
import string
import shutil
import argparse
import subprocess
import numpy as np
import thualign.data as data
import thualign.models as models
import thualign.utils as utils
import thualign.utils.alignment as alignment_utils
from datetime import datetime
import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test neural alignment models",
        usage="inferrer.py [<args>] [-h | --help]"
    )
    # test args
    parser.add_argument("--gen-weights", action="store_true", help="whether to generate attention weights")
    parser.add_argument("--gen-vizdata", action="store_true", help="whether to generate visualization data")
    parser.add_argument("--test-aer", action="store_true", help="whether to test aer for alignments")

    # configure file
    parser.add_argument("--config", type=str, required=True,
                        help="Provided config file")
    parser.add_argument("--base-config", type=str, help="base config file")
    parser.add_argument("--data-config", type=str, help="data config file")
    parser.add_argument("--model-config", type=str, help="base config file")
    parser.add_argument("--exp", "-e", default='DEFAULT', type=str, help="name of experiments")

    return parser.parse_args()


def load_vocabulary(params):
    params.vocabulary = {
        "source": data.Vocabulary(params.vocab[0]), 
        "target": data.Vocabulary(params.vocab[1])
    }
    return params

def to_cuda(features):
    for key in features:
        features[key] = features[key].cuda()

    return features

def gen_weights(params):
    """Generate attention weights 
    """
    
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
    dist.init_process_group("nccl", init_method=url,
                            rank=0,
                            world_size=1)

    params = load_vocabulary(params)
    checkpoint = getattr(params, "checkpoint", None) or utils.best_checkpoint(params.output)
    # checkpoint = getattr(params, "checkpoint", None) or utils.latest_checkpoint(params.output)

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    test_path = params.test_path
    # Create directory and copy files
    if not os.path.exists(test_path):
        print("Making dir: %s" % test_path)
        os.makedirs(test_path)
        params_pattern = os.path.join(params.output, "*.config")
        params_files = glob.glob(params_pattern)

        for name in params_files:
            new_name = name.replace(params.output, test_path)
            shutil.copy(name, new_name)

    if params.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Create model
    with torch.no_grad():

        model = models.get_model(params).cuda()

        if params.half:
            model = model.half()
        
        model.eval()
        print('loading checkpoint: {}'.format(checkpoint))
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

        # dataset = data.get_dataset(params, "infer")

        get_infer_dataset = data.AlignmentPipeline.get_infer_dataset
        dataset = get_infer_dataset(params.test_input, params)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=None)

        # text data
        src = [l.strip().split() for l in open(params.test_input[0])]
        tgt = [l.strip().split() for l in open(params.test_input[1])]


        iterator = iter(dataset)
        counter = 0
        pad_max = 1024

        f_cross_attn = []
        b_cross_attn = []

        line_start = 0
        line_end = 0

        weight_path = getattr(params, 'weight_path', None) or params.test_path
        output_path = getattr(params, 'output_path', None) or weight_path
        output_name = os.path.join(output_path, 'alignment.txt')
        cur_len = 0
        cur_lines = []
        if os.path.exists(output_name):
            cur_lines = open(output_name, "r").readlines()
            cur_len = len(cur_lines)
            save_batch_num = int(cur_len / params.decode_batch_size)
            new_f = open(output_name, "w")
            new_f.writelines(cur_lines[:save_batch_num * params.decode_batch_size])
            new_f.close()
        extract_params = alignment_utils.get_extract_params(params)
        with open(output_name, 'a+') as align_out:
            while True:
                try:
                    features = next(iterator)
                    features = to_cuda(features)
                except:
                    break

                # t = time.time()
                counter += 1
                if counter * params.decode_batch_size <= cur_len:
                    line_start += params.decode_batch_size
                    print("Have Finished batch(%d)!" % (counter))
                    continue
                # Decode
                acc_cnt, all_cnt, state = model.cal_alignment(features)

                # t = time.time() - t

                # score = 0.0 if all_cnt == 0 else acc_cnt / all_cnt
                print("Finished batch(%d): %s" % (counter, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    
                
                source_lengths, target_lengths = features["source_mask"].sum(-1).long().tolist(), features["target_mask"].sum(-1).long().tolist()
                line_end = line_start + len(source_lengths)
                
                for src_i, tgt_i, weight_f, weight_b, src_len, tgt_len in zip(src[line_start:line_end], tgt[line_start:line_end], state['f_cross_attn'], state['b_cross_attn'], source_lengths, target_lengths):
                    weight_f, weight_b = weight_f[:, :, :tgt_len, :src_len][-1].mean(dim=0), weight_b[:, :, :tgt_len, :src_len][-1].mean(dim=0)
                    align_soft, weight_final = alignment_utils.bidir_weights_to_align(weight_f, weight_b, src_i, tgt_i, **extract_params)
                    align_out.write(str(align_soft) + '\n')
                line_start = line_end
                # old version
                # for weight_f, weight_b, src_len, tgt_len in zip(state['f_cross_attn'], state['b_cross_attn'], source_lengths, target_lengths):
                #     f_cross_attn.append(weight_f[:, :, :tgt_len, :src_len][-1].mean(dim=0))
                #     b_cross_attn.append(weight_b[:, :, :tgt_len, :src_len][-1].mean(dim=0))                

                # line_end = line_start + len(f_cross_attn)
                # gen_align(params, src[line_start:line_end], tgt[line_start:line_end], f_cross_attn, b_cross_attn, align_out)
                # line_start = line_end
                # f_cross_attn, b_cross_attn = [], []


def gen_align(params, src, tgt, f_weights, b_weights, align_out):
    """Generate alignment
    """

    extract_params = alignment_utils.get_extract_params(params)
 
    # bidirectional
    hyps = {} # forward, backward, gdf, soft-extraction
    hyps["soft"] = []

    for i in range(len(f_weights)):
        src_i, tgt_i, weight_f, weight_b = src[i], tgt[i], f_weights[i], b_weights[i]
        # weight_f, weight_b = weight_f[-1].mean(dim=0), weight_b[-1].mean(dim=0)

        # soft extraction
        align_soft, weight_final = alignment_utils.bidir_weights_to_align(weight_f, weight_b, src_i, tgt_i, **extract_params)
        align_out.write(str(align_soft) + '\n')
            

def merge_dict(src, tgt, keep_common=True):
    res = {}
    for k, v in src.items():
        res[k] = v
    for k, v in tgt.items():
        if k not in res:
            res[k] = v
        elif keep_common:
            src_k = k + '_f'
            tgt_k = k + '_b'
            res[src_k] = res.pop(k)
            res[tgt_k] = v
    return res

def merge(forward_vizdata, backward_vizdata, bialigns):
    res = []
    assert len(forward_vizdata) == len(backward_vizdata) == len(bialigns)
    for f, b, bialign in zip(forward_vizdata, backward_vizdata, bialigns):
        res_t = {}
        assert f['src'] == b['src'] and f['tgt'] == b['tgt']
        res_t['src'] = f['src']
        res_t['tgt'] = f['tgt']
        res_t['weights'] = merge_dict(f['weights'], b['weights'])
        res_t['metrics'] = merge_dict(f['metrics'], b['metrics'])
        res_t['weights']['bidir'] = alignment_utils.align_to_weights(bialign, bialign, f['src'], f['tgt'])

        if 'ref' in f and 'ref' in b:
            assert f['ref'] == b['ref']
            ref_t, pos_t = alignment_utils.parse_ref(f['ref'])
            metric_t = alignment_utils.alignment_metrics([bialign], [ref_t], [pos_t])
            res_t['metrics']['bidir'] = metric_t
        res.append(res_t)
    return res

def main(args):
    exps = args.exp.split(',')
    exp_params = []

    for exp in exps:
        params = utils.Config.read(args.config, base=args.base_config, data=args.data_config, model=args.model_config, exp=exp)
        exp_params.append(params)
        
        base_dir = params.output
        test_path = os.path.join(base_dir, "test")
        params.test_path = test_path
        params.test_path = os.path.join(params.align_output, "align_out")
        
        params.gen_weights = args.gen_weights
        params.gen_vizdata = args.gen_vizdata
        params.test_aer = args.test_aer

        if params.gen_weights:
            gen_weights(params)
        # gen_align(params)

    if len(exps) == 2:
        import thualign.scripts.combine_bidirectional_alignments
        import thualign.scripts.aer

        def infer_test_path(params):
            weight_path = getattr(params, 'weight_path', None) or params.test_path
            output_path = getattr(params, 'output_path', None) or weight_path
            return output_path

        test_paths = [infer_test_path(params) for params in exp_params]
        common_path = os.path.commonpath(test_paths)
        output_alignment_file = os.path.join(common_path, 'alignment.txt')

        # combine bidirectional alignments
        completedProcess = subprocess.run('python {script} {alignments} --dont_reverse --method grow-diagonal > {output}'.format(
            script=thualign.scripts.combine_bidirectional_alignments.__file__,
            alignments=' '.join([os.path.join(test_path, 'alignment.txt') for test_path in test_paths]),
            output=output_alignment_file
        ), shell=True)

        # calculate aer for combined alignments
        completedProcess = subprocess.run('python {script} {ref} {alignment} > {aer_res}'.format(
            script=thualign.scripts.aer.__file__,
            ref=exp_params[0].test_ref,
            alignment=output_alignment_file,
            aer_res=os.path.join(common_path, 'aer_res.txt')
        ), shell=True)

        print(open(os.path.join(common_path, 'aer_res.txt')).read())

        merge_data_flag = True
        vizdata_files = [os.path.join(test_path, 'alignment_vizdata.pt') for test_path in test_paths]
        for vizdata_file in vizdata_files:
            merge_data_flag = merge_data_flag and os.path.exists(vizdata_file)

        if merge_data_flag:
            forward_vizdata = torch.load(vizdata_files[0], map_location='cpu') # list of ['src', 'tgt', 'ref', 'weights', 'metrics']
            backward_vizdata = torch.load(vizdata_files[1], map_location='cpu') # list of ['src', 'tgt', 'ref', 'weights', 'metrics']
            bialigns = [alignment_utils.parse_ref(line.strip())[0] for line in open(output_alignment_file)]
            merged_vizdata = merge(forward_vizdata, backward_vizdata, bialigns)
            torch.save(merged_vizdata, os.path.join(common_path, 'alignment_vizdata.pt'))
            

    if len(exps) > 2:
        print('More than two experiments are not supported! No merging action will be performed.')

if __name__ == "__main__":
    main(parse_args())