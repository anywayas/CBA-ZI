# -*- coding: utf-8 -*-

import torch
import random
import os
import json
import argparse
import pandas as pd
import numpy as np
from sub_edge_embedding import DrugDataset
from sub_edge_embedding import DrugDataLoader
from models import RealFakeDDICo
from engine4rf import Engine4RealFakeDDI

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--end_lr", type=float, default=1e-6)
    parser.add_argument("--lr_decay_interval", type=int, default=3000)
    parser.add_argument("--lr_decay_rate", type=float, default=0.95)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ratio_k", type=int, default=26)
    #data 3
    parser.add_argument("--train_filename", type=str,default='data/drugbank_cold_start/3/ddi_train_cleaned.csv')
    #new-new
    parser.add_argument("--test_filename1", type=str,default='data/drugbank_cold_start/3/ddi_bothnew_cleaned.csv')
    #new-old
    parser.add_argument("--test_filename2", type=str,default='data/drugbank_cold_start/3/ddi_eithernew_cleaned.csv')
    parser.add_argument("--sample_file", type=str, default='data/drugbank_cold_start/3/ssi_old_new_3.json')
    parser.add_argument("--block_path", type=str, default='checkpoint/3/nn_acc0.7144roc0.7779prc0.7939dp0.2k26bs256')
    parser.add_argument("--fine_model_path", default='checkpoint/3')
    parser.add_argument("--test", default=False)
    args = parser.parse_args()
    return args


def set_all_seeds(seed): #6535
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def worker_init(worker_init):
    seed= 6535
    np.random.seed(int(seed) + worker_init)

def get_data(args,drug_old_ids, drug_new_ids):
    # train
    df_ddi_train = pd.read_csv(args.train_filename)
    train_tup = [(h, t, r, 1) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
    train_data = DrugDataset(train_tup, all_drug_old_ids=drug_old_ids, all_drug_new_ids=drug_new_ids, mode="ttrain")
    # test(new-new)
    df_ddi_test = pd.read_csv(args.test_filename1)
    test_tup = [(d1, d2, t, 1) for d1, d2, t in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]
    test_neg_tup = train_data.generate_neg_pair(test_tup)
    test_tup = test_tup + test_neg_tup
    test_data1 = DrugDataset(test_tup,all_drug_old_ids=drug_old_ids, all_drug_new_ids=drug_new_ids, mode="ttest")
    # test(new-old)
    df_ddi_test = pd.read_csv(args.test_filename2)
    test_tup = [(d1, d2, t, 1) for d1, d2, t in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]
    test_neg_tup = train_data.generate_neg_pair(test_tup)
    test_tup = test_tup + test_neg_tup
    test_data2 = DrugDataset(test_tup,all_drug_old_ids=drug_old_ids, all_drug_new_ids=drug_new_ids, mode="ttest")

    print(f"Training with {args.train_filename}, test with {args.test_filename1},{args.test_filename2}")
    print(f"Training with {len(train_data)} samples, new-new test with {len(test_data1)} samples")
    print(f"New-old test with {len(test_data2)} samples")
    return train_data,test_data1,test_data2


def main():
    set_all_seeds(6535)
    torch.autograd.set_detect_anomaly(True)

    args = config()

    with open(args.sample_file, 'r', encoding='utf8') as f:
        ontology = json.load(f)
    drug_old_ids = np.array(ontology['old'])
    drug_new_ids = np.array(ontology['new'])
    train_data,test_data1,test_data2 = get_data(args, drug_old_ids, drug_new_ids)

    train_dataloader = DrugDataLoader(train_data,
                                      collate_fn=train_data.collate_fn_fake_or_real,
                                      batch_size=args.batch_size,
                                      worker_init_fn=worker_init,
                                      shuffle=True,
                                      num_workers = 3)

    test_dataloader1 = DrugDataLoader(test_data1,
                                      collate_fn=test_data1.collate_fn_fake_or_real,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=3)

    test_dataloader2 = DrugDataLoader(test_data2,
                                     collate_fn=test_data2.collate_fn_fake_or_real,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=3)
    if args.device >= 0:
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu:{}".format(0))

    model = RealFakeDDICo(with_global=True, ratio_k=args.ratio_k,  dropout_ratio=args.dropout)
    model = model.to(device)

    assert 256 % args.batch_size == 0
    gradient_stack = None
    if args.batch_size < 256:
        gradient_stack = 256 // args.batch_size

    engine = Engine4RealFakeDDI(args,
                                model,
                                train_dataloader,
                                test_dataloader1,
                                test_dataloader2,
                                None,
                                args.end_lr,
                                args.lr_decay_interval,
                                args.lr_decay_rate,
                                args.fine_model_path,
                                device,
                                gradient_stack=gradient_stack)

    if args.test:
        print("load pretrained graph model")
        model.load_state_dict(torch.load(args.block_path))
        print("new-new test:")
        engine.test(test_dataloader1, If_newnew=True)
        print("new-old test:")
        engine.test(test_dataloader2, If_newnew=False)
    else:
        engine.train(args.epoch)


if __name__ == '__main__':
    main()

