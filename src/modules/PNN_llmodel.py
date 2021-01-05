import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.ll_model import LifelongLearningModel


class PNNLinearBlock(nn.Module):
    def __init__(self, in_sizes, out_size, scalar_mult=1.0, split_v=False):
        super(PNNLinearBlock, self).__init__()
        assert isinstance(in_sizes, (list, tuple))
        self.split_v = split_v
        self.in_sizes = in_sizes
        self.w = nn.Linear(in_sizes[-1], out_size)

        if len(in_sizes) > 1:
            self.alphas = nn.ParameterList()
            for i in range(len(in_sizes) - 1):
                new_alpha = torch.tensor([scalar_mult])#.expand(in_size)
                self.alphas.append(nn.Parameter(new_alpha))
            if split_v:
                self.v = nn.ModuleList([nn.Linear(in_size, in_sizes[-1]) for in_size in in_sizes[:-1]])
            else:
                v_in_size = sum(in_sizes[:-1])
                self.v = nn.Linear(v_in_size, in_sizes[-1])
            self.u = nn.Linear(in_sizes[-1], out_size)

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out = self.w(inputs[-1])
        if len(inputs) > 1:
            prev_columns_out = []
            for x, alpha in zip(inputs, self.alphas):
                prev_columns_out.append(alpha * x)
            if self.split_v:
                prev_columns_out = [v(x) for v, x in zip(self.v, prev_columns_out)]
                prev_columns_out = sum(prev_columns_out)
            else:
                prev_columns_out = torch.cat(prev_columns_out, dim=1)
                prev_columns_out = self.v(prev_columns_out)
            out += self.u(F.relu(prev_columns_out))
        return out


class PNN(nn.Module):
    def __init__(self, columns):
        super(PNN, self).__init__()
        self.columns = columns

    def forward(self, x, task_id=-1):
        assert self.columns, 'PNN should at least have one column ' \
                             '(missing call to `new_task` ?)'
        inputs = [c[0](x) for c in self.columns]

        for l in range(1, len(self.columns[0])):
            inputs = list(map(F.relu, inputs))
            outputs = []

            #TODO: Use task_id to check if all columns are necessary
            for i, column in enumerate(self.columns):
                outputs.append(column[l](inputs[:i+1]))
            inputs = outputs
        return inputs[task_id]

    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            for params in c.parameters():
                params.requires_grad = i in skip


    # def parameters(self, col=-1):
    #     # if col is None:
    #     #     return super(PNN, self).parameters()
    #
    #     return self.columns[col].parameters()


class PNN_LLmodel(LifelongLearningModel):
    def __init__(self, split_v, *args, **kwargs):
        super(PNN_LLmodel, self).__init__(*args, **kwargs)
        self.split_v = split_v
        self.columns = []

    def _new_model(self, x_dim, n_classes, task_id, **kwargs):
        msg = "Should have the out size for each layer + input size " \
              "(got {} sizes but {} layers)."
        sizes = self.get_sizes(x_dim, n_classes)
        # assert len(sizes) == self.n_layers + 1, msg.format(len(sizes),
        #                                                        self.n_layers)
        assert task_id == len(self.columns)

        new_column = nn.ModuleList([])
        new_column.append(PNNLinearBlock([sizes[0]], sizes[1],
                                         split_v=self.split_v))
        for i in range(1, len(sizes)-1):
            new_column.append(
                PNNLinearBlock([sizes[i]] * (task_id + 1), sizes[i + 1],
                               split_v=self.split_v))
        self.columns.append(new_column)
        model = PNN(nn.ModuleList(self.columns))
        model.n_out = 1
        return model

    def get_model(self, task_id, *args, **kwargs):
        model = super().get_model(task_id, *args, **kwargs)
        assert len(model.columns) == task_id + 1
        model.freeze_columns(skip=[task_id])
        return model

    def finish_task(self, dataset, task_id, viz=None):
        self.get_model(task_id).freeze_columns()