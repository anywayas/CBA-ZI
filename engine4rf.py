# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from eval import real_fake_eval

class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()

    def forward(self, rel_logit, rel_labels):
        """
        shape of p_score, ground_truth:  tensor:(batchsize*2,1)
        """
        #rel_logit = rel_logit.squeeze(1)
        batchsize = int(len(rel_logit)/2)
        p_scores = rel_logit[:batchsize]
        n_scores = rel_logit[batchsize:]
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        return (p_loss + n_loss) / 2


class Engine4RealFakeDDI:
    def __init__(self,
                 args,
                 model,
                 train_loader: DataLoader,
                 test_loader1: DataLoader,
                 test_loader2: DataLoader,
                 scheduler,
                 end_lr,
                 lr_decay_interval,
                 lr_decay_rate,
                 fine_model_path,
                 device,
                 gradient_stack=None):

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.end_lr = end_lr
        self.lr_decay_interval = lr_decay_interval
        self.lr_decay_rate = lr_decay_rate
        self.loss = SigmoidLoss()
        self.train_loader = train_loader
        self.test_loader1 = test_loader1
        self.test_loader2 = test_loader2
        self.device = device
        self.fine_model_path = fine_model_path
        self.dropout = args.dropout
        self.k = args.ratio_k
        self.bs = args.batch_size

        if gradient_stack is not None:
            assert isinstance(gradient_stack, int)
            assert gradient_stack >= 1

        self.gradient_stack = gradient_stack

    def test(self, dataloader, If_newnew):
        #test_bar = tqdm(dataloader)
        test_bar = dataloader
        all_rel_probs = []
        all_rel_labels = []
        with torch.no_grad():
            self.model.eval()
            for batch_tensors in test_bar:
                batch_tensors = [tensor.to(device=self.device) for tensor in batch_tensors]
                # batch_tensors = [torch.tensor(tensor, device=self.device) for tensor in batch_tensors]
                (h_data, t_data, rels, label) = batch_tensors

                rets = self.model.forward(h_data, t_data, rels)
                rel_logit, rel_probs = rets
                all_rel_probs.append(rel_probs.cpu().numpy())
                all_rel_labels.append(label.cpu().numpy())
            self.model.train()
            all_rel_probs = np.concatenate(all_rel_probs, 0)
            all_rel_labels = np.concatenate(all_rel_labels, 0)
            acc, auc, ap, precision, recall, f1, old_acc = real_fake_eval(all_rel_probs, all_rel_labels)
            print("acc:{:.4f}, auc:{:.4f}, ap:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}, optimal_acc:{:.4f}" \
                  .format(acc, auc, ap, precision, recall, f1, old_acc))

            #save model
            if (If_newnew == True):
                if (acc > 0.68):
                    path = self.fine_model_path + "/nn_acc{:.4f}roc{:.4f}prc{:.4f}dp{:.1f}k{:d}bs{:d}".format(acc,
                                                                                                              auc_roc,
                                                                                                              auc_prc,
                                                                                                              self.dropout,
                                                                                                              self.k,
                                                                                                              self.bs)
                    self.save(path)
            elif (If_newnew == False):
                if (acc > 0.76):
                    path = self.fine_model_path + "/no_acc{:.4f}roc{:.4f}prc{:.4f}dp{:.1f}k{:d}bs{:d}".format(acc,
                                                                                                              auc_roc,
                                                                                                              auc_prc,
                                                                                                              self.dropout,
                                                                                                              self.k,
                                                                                                              self.bs)
                    self.save(path)

    def train(self, epoch_num):
        global_step = 0
        self.model.train()
        self.optimizer.zero_grad()

        for e in range(epoch_num):
            train_bar = tqdm(enumerate(self.train_loader))
            avg_loss= 0
            batch_num = 0
            for step, batch_tensors in train_bar:
                # batch_tensors = [torch.tensor(tensor, device=self.device) for tensor in batch_tensors]
                batch_tensors = [tensor.to(device=self.device) for tensor in batch_tensors]
                (h_data, t_data, rels, label) = batch_tensors
                global_step += 1
                rets = self.model.forward(h_data, t_data, rels)
                rel_logit, rel_probs = rets

                ce = self.loss(rel_logit, label)
                ce_float = ce.item()
                avg_loss += ce_float
                batch_num = batch_num + 1

                if self.gradient_stack is not None:
                    ce = ce / float(self.gradient_stack)
                ce.backward()

                update_grad = False
                if global_step % (self.gradient_stack or 1) == 0:
                    nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    update_grad = True
                train_bar.set_description("Epoch:{} Step:{} Global:{} Loss:{:.4f} Update_gradient: {}".format(e,
                                                                                                              step + 1,
                                                                                                              global_step,
                                                                                                              ce_float,
                                                                                                              update_grad))
                if global_step % (self.lr_decay_interval * (self.gradient_stack or 1)) == 0:
                    self.update_lr()

                if (step + 1) % (500 * (self.gradient_stack or 1)) == 0 and e >= 8:
                    print("test metrics in {:d}".format(step))
                    print("loss : {:.4f} ".format(ce.item()))
                    self.test(self.test_loader1, If_newnew=True)
                    self.test(self.test_loader2, If_newnew=False)

            avg_loss = avg_loss / batch_num
            print("Epoch: {:d}  loss : {:.4f} ".format(e, avg_loss))

            # test after one epoch training
            self.test(self.test_loader1, If_newnew=True)
            self.test(self.test_loader2, If_newnew=False)

    def update_lr(self):
        for pg in self.optimizer.param_groups:
            pg['lr'] = max(self.end_lr, pg['lr'] * self.lr_decay_rate)

    def save(self,path):
        torch.save(self.model.state_dict(), path)
