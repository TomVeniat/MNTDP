import logging

import torch

import numpy as np
from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import get_conv_out_size
from src.utils.misc import count_params

logger = logging.getLogger(__name__)


class MLPHAT(torch.nn.Module):

    def __init__(self,inputsize, clipgrad, thres_cosh, thres_emb, wide,
                 nlayers=2,nhid=1000,pdrop1=0.2,pdrop2=0.5):
        super(MLPHAT,self).__init__()

        ncha,size,_=inputsize
        self.wide = wide

        self.nlayers=nlayers
        self.nhid = int(nhid * self.wide)

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)
        self.fc1=torch.nn.Linear(ncha*size*size, self.nhid)
        if nlayers>1:
            self.fc2=torch.nn.Linear(self.nhid, self.nhid)
        self.last=torch.nn.ModuleList()

        self.gate=torch.nn.Sigmoid()
        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""
        self.embeddings = None
        self.iter = None
        self.iter_per_epoch = None
        self.n_out = 1
        self.cur_task_id = None
        self.clipgrad = clipgrad
        self.thres_cosh = thres_cosh
        self.thres_emb = thres_emb

    def expand_embeddings(self):
        if self.embeddings is None:
            efc1 = torch.nn.Embedding(1, self.nhid)
            efc2 = torch.nn.Embedding(1, self.nhid)
            embeddings = [efc1, efc2]
            self.embeddings = torch.nn.ModuleList(embeddings)

            return
        updated_embeddings = torch.nn.ModuleList()
        for emb in self.embeddings:
            w = emb.weight
            new_emb = torch.nn.Embedding(1, w.shape[1])
            updated_w = torch.cat([w, new_emb.weight], )
            updated_embeddings.append(
                torch.nn.Embedding.from_pretrained(updated_w, freeze=False)
            )
        self.embeddings = updated_embeddings

    def add_head(self, n_classes):
        self.last.append(torch.nn.Linear(self.nhid, n_classes))

    def forward(self, x):
        if self.training:
            self.s = (self.smax - 1 / self.smax) * self.iter / self.iter_per_epoch \
                     + 1 / self.smax
        else:
            self.s = self.smax
        # Gates
        masks = self.mask(torch.LongTensor([self.cur_task_id]).to(x.device),
                        s=self.s)
        gfc1, gfc2 = masks
        # Gated
        h=self.drop1(x.view(x.size(0),-1))
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)

        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        y = self.last[self.cur_task_id](h)
        return y,masks

    def mask(self,t,s=1):
        return [self.gate(s*e(t)) for e in self.embeddings]

    def get_view_for(self,n,masks):
        gfc1,gfc2=masks
        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)

        return None

    def new_iter_hook(self, *args):
        self.iter += 1

    def new_epoch_hook(self, *args):
        self.iter = 0

    def arch_repr(self):
        p = {}
        p[0] = count_params(self.fc1)['trainable']
        p[1] = count_params(self.fc2)['trainable']
        p[2] = count_params(self.last)['trainable']
        return p

    def loss_wrapper(self, loss_fn):
        def loss(y_pred, y, **kwargs):
            y_pred, masks = y_pred
            task_loss = loss_fn(y_pred, y)
            # new_loss = lambda y_hat, y: F.cross_entropy(y_hat, y.squeeze()) + self.model.ewc_loss()
            return task_loss + self.reg_loss(masks)

        return loss

    def reg_loss(self, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1 - mp.to(m.device)
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count
        return self.lamda * reg

    def post_backward_hook(self):
        if self.cur_task_id > 0:
            for n, p in self.named_parameters():
                if n in self.mask_back:
                    p.grad.data *= self.mask_back[n].to(p.device)

        # Compensate embedding gradients
        for n, p in self.named_parameters():
            if n.startswith('e'):
                num = torch.cosh(
                    torch.clamp(self.s * p.data, -self.thres_cosh, self.thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.smax / self.s * num / den

            # Apply step
        torch.nn.utils.clip_grad_norm_(self.parameters(),
                                       self.clipgrad)

    def post_update_hook(self):
        for n, p in self.named_parameters():
            if n.startswith('e'):
                p.data = torch.clamp(p.data, -self.thres_emb, self.thres_emb)

class AlexnetHAT(torch.nn.Module):

    def __init__(self, inputsize, clipgrad, thres_cosh, thres_emb, wide):
        super(AlexnetHAT, self).__init__()

        ncha, size, _ = inputsize
        self.wide = wide
        self.c1 = torch.nn.Conv2d(ncha, int(64*wide), kernel_size=size // 8)
        s = get_conv_out_size([size], size // 8, 0, 1)[0]
        s = s // 2
        self.c2 = torch.nn.Conv2d(int(64*wide), int(128*wide), kernel_size=size // 10)
        s = get_conv_out_size([s], size // 10, 0, 1)[0]
        s = s // 2
        self.c3 = torch.nn.Conv2d(int(128*wide), int(256*wide), kernel_size=2)
        s = get_conv_out_size([s], 2, 0, 1)[0]
        s = s // 2
        self.smid = s
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(int(256 * wide) * self.smid * self.smid, int(2048*wide))
        self.fc2 = torch.nn.Linear(int(2048 * wide), int(2048 * wide))
        self.last = torch.nn.ModuleList()

        self.gate = torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.embeddings = None
        self.iter = None
        self.iter_per_epoch = None
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""
        self.n_out = 1
        self.cur_task_id = None
        self.clipgrad = clipgrad
        self.thres_cosh = thres_cosh
        self.thres_emb = thres_emb

    def expand_embeddings(self):
        if self.embeddings is None:
            ec1 = torch.nn.Embedding(1, int(64 * self.wide))
            ec2 = torch.nn.Embedding(1, int(128 * self.wide))
            ec3 = torch.nn.Embedding(1, int(256 * self.wide))
            efc1 = torch.nn.Embedding(1, int(2048 * self.wide))
            efc2 = torch.nn.Embedding(1, int(2048 * self.wide))
            embeddings = [ec1, ec2, ec3, efc1, efc2]
            self.embeddings = torch.nn.ModuleList(embeddings)

            return
        updated_embeddings = torch.nn.ModuleList()
        for emb in self.embeddings:
            w = emb.weight
            new_emb = torch.nn.Embedding(1, w.shape[1])
            updated_w = torch.cat([w, new_emb.weight], )
            updated_embeddings.append(
                torch.nn.Embedding.from_pretrained(updated_w, freeze=False)
            )
        self.embeddings = updated_embeddings

    def add_head(self, n_classes):
        self.last.append(torch.nn.Linear(int(2048*self.wide), n_classes))

    def forward(self, x):
        # Gates
        if self.training:
            self.s = (self.smax - 1 / self.smax) * self.iter / self.iter_per_epoch \
                + 1 / self.smax
        else:
            self.s = self.smax
        masks = self.mask(torch.LongTensor([self.cur_task_id]).to(x.device),
                          s=self.s)
        gc1, gc2, gc3, gfc1, gfc2 = masks
        # Gated
        h = self.maxpool(self.drop1(self.relu(self.c1(x))))
        h = h * gc1.view(1, -1, 1, 1).expand_as(h)
        h = self.maxpool(self.drop1(self.relu(self.c2(h))))
        h = h * gc2.view(1, -1, 1, 1).expand_as(h)
        h = self.maxpool(self.drop2(self.relu(self.c3(h))))
        h = h * gc3.view(1, -1, 1, 1).expand_as(h)
        h = h.view(x.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = h * gfc1.expand_as(h)
        h = self.drop2(self.relu(self.fc2(h)))
        h = h * gfc2.expand_as(h)
        y = self.last[self.cur_task_id](h)
        return y, masks

    def mask(self, t, s=1):
        return [self.gate(s*e(t)) for e in self.embeddings]

    def get_view_for(self, n, masks):
        ec1, ec2, ec3, efc1, efc2 = self.embeddings
        gc1, gc2, gc3, gfc1, gfc2 = masks
        if n == 'fc1.weight':
            post = gfc1.data.view(-1, 1).expand_as(self.fc1.weight)
            pre = gc3.data.view(-1, 1, 1).expand((ec3.weight.size(1),
                                                  self.smid,
                                                  self.smid)).contiguous().view(
                1, -1).expand_as(self.fc1.weight)
            return torch.min(post, pre)
        elif n == 'fc1.bias':
            return gfc1.data.view(-1)
        elif n == 'fc2.weight':
            post = gfc2.data.view(-1, 1).expand_as(self.fc2.weight)
            pre = gfc1.data.view(1, -1).expand_as(self.fc2.weight)
            return torch.min(post, pre)
        elif n == 'fc2.bias':
            return gfc2.data.view(-1)
        elif n == 'c1.weight':
            return gc1.data.view(-1, 1, 1, 1).expand_as(self.c1.weight)
        elif n == 'c1.bias':
            return gc1.data.view(-1)
        elif n == 'c2.weight':
            post = gc2.data.view(-1, 1, 1, 1).expand_as(self.c2.weight)
            pre = gc1.data.view(1, -1, 1, 1).expand_as(self.c2.weight)
            return torch.min(post, pre)
        elif n == 'c2.bias':
            return gc2.data.view(-1)
        elif n == 'c3.weight':
            post = gc3.data.view(-1, 1, 1, 1).expand_as(self.c3.weight)
            pre = gc2.data.view(1, -1, 1, 1).expand_as(self.c3.weight)
            return torch.min(post, pre)
        elif n == 'c3.bias':
            return gc3.data.view(-1)
        return None

    def new_iter_hook(self, *args):
        self.iter += 1

    def new_epoch_hook(self, *args):
        self.iter = 0

    def arch_repr(self):
        p = {}
        p[0] = count_params(self.c1)['trainable']
        p[1] = count_params(self.c2)['trainable']
        p[2] = count_params(self.c3)['trainable']
        p[3] = count_params(self.fc1)['trainable']
        p[4] = count_params(self.fc2)['trainable']
        p[5] = count_params(self.last)['trainable']
        return p

    def loss_wrapper(self, loss_fn):
        def loss(y_pred, y, **kwargs):
            y_pred, masks = y_pred
            task_loss = loss_fn(y_pred, y)
            # new_loss = lambda y_hat, y: F.cross_entropy(y_hat, y.squeeze()) + self.model.ewc_loss()
            return task_loss + self.reg_loss(masks)

        return loss

    def reg_loss(self, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1 - mp.to(m.device)
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count
        return self.lamda * reg

    def post_backward_hook(self):
        if self.cur_task_id > 0:
            for n, p in self.named_parameters():
                if n in self.mask_back:
                    p.grad.data *= self.mask_back[n].to(p.device)

        # Compensate embedding gradients
        for n, p in self.named_parameters():
            if n.startswith('e'):
                num = torch.cosh(
                    torch.clamp(self.s * p.data, -self.thres_cosh, self.thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.smax / self.s * num / den

            # Apply step
        torch.nn.utils.clip_grad_norm_(self.parameters(),
                                       self.clipgrad)

    def post_update_hook(self):
        for n, p in self.named_parameters():
            if n.startswith('e'):
                p.data = torch.clamp(p.data, -self.thres_emb, self.thres_emb)


class HATLLModel(LifelongLearningModel):
    def __init__(self, clipgrad, thres_cosh, thres_emb, wide, mode, *args, **kwargs):
        super(HATLLModel, self).__init__(*args, **kwargs)
        self.clipgrad = clipgrad
        self.thres_cosh = thres_cosh
        self.thres_emb = thres_emb
        self.lamda = None
        self.smax = None
        self.wide = wide
        self.mode = mode

        self.model = None
        self.mask_pre = None
        self.mask_back = None

        if wide is False:
            wide = 1
        assert wide >= 1
        self.wide = wide

    def set_h_params(self, lamda, smax):
        self.lamda = lamda
        self.smax = smax

    def _new_model(self, x_dim, n_classes, task_id, **kwargs):
        if self.model is None:
            assert task_id == 0
            if self.mode == 'cnn':
                mod = AlexnetHAT
            elif self.mode == 'mlp':
                mod = MLPHAT
            else:
                raise ValueError(f'Unknown mode {self.mode}')
            self.model = mod(x_dim, self.clipgrad, self.thres_cosh,
                                    self.thres_emb, self.wide)

        assert task_id == len(self.model.last)
        self.model.expand_embeddings()
        self.model.add_head(n_classes[0])
        return self.model

    def get_model(self, task_id, **task_infos):
        if self.model is None or task_id >= len(self.model.last):
            assert 'x_dim' in task_infos and 'n_classes' in task_infos
            self._new_model(task_id=task_id, **task_infos)
        model = self.model
        model.cur_task_id = task_id
        model.lamda = self.lamda
        model.smax = self.smax
        model.mask_pre = self.mask_pre
        model.mask_back = self.mask_back
        # model.register_buffer('mask_pre', self.mask_pre)
        return model

    def finish_task(self, dataset, task_id, viz=None, path=None):
        device = next(iter(self.model.parameters())).device
        task = torch.LongTensor([task_id]).to(device)
        mask = self.model.mask(task, s=self.smax)
        for i in range(len(mask)):
            mask[i] = mask[i].detach().clone()
        if task_id == 0:
            self.mask_pre = mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i] = torch.max(self.mask_pre[i].to(device),
                                             mask[i])

        # Weights mask
        self.mask_back = {}
        for n, _ in self.model.named_parameters():
            vals = self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals
        return {}






