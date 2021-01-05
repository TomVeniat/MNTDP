import torch
from ignite.engine import Engine
from ignite.utils import convert_tensor


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y, *z = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
            *z)


def out(x, y, y_pred, loss, optionals):
    return x, y, y_pred, loss, optionals


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              # output_transform=lambda x, y, y_pred, loss: loss.item()):
                              # output_transform=lambda x, y, y_pred, loss: (x, y, y_pred, loss)):
                              output_transform=out):
    """
    From Ignite
    """
    if device:
        model.to(device)

    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer = [optimizer]

    def _update(engine, batch):
        model.train()
        for opt in optimizer:
            opt.zero_grad()
        # z is optional (e.g. task ids)
        x, y, *z = prepare_batch(batch,
                                 device=device,
                                 non_blocking=non_blocking)
        y_pred = model(*(x, *z))
        loss = loss_fn(y_pred, y)
        if torch.is_tensor(loss):
            details = {}
        else:
            loss, details = loss
        loss = loss.mean()

        if hasattr(model, 'cur_split') and model.cur_split is not None:
            # If the model specifies a specific train split, select
            # corresponding optimizers
            selected_optims = [optimizer[model.cur_split]]
        else:
            # Otherwise select them all
            selected_optims = optimizer

        if not hasattr(model, 'requires_grad') or model.requires_grad():
            loss.backward()
            # params = [opt.param_groups[0]['params'][0].grad for opt in optimizer]
            # aa_grad = [(p ** 2).mean() if p is not None else p for p in params]
            if hasattr(model, 'post_backward_hook'):
                model.post_backward_hook()
            for opt in selected_optims:
                opt.step()
            if hasattr(model, 'post_update_hook'):
                model.post_update_hook()

        if hasattr(model, 'arch_sampler'):
            ent_mean = model.global_entropy()
            ent_sampling = model.sampled_entropy()
            arch_grad = [p.abs() for p in model.arch_sampler.parameters()]
            grad_dict = {k: arch_grad[v].detach()
                          for k, v in model.arch_sampler.group_ids_index.items()}
            probas = model.arch_sampler().squeeze()
            proba_dict = {n: probas[id] for n, id in model.ssn.stochastic_node_ids.items()}
            node_grads = {}
            for params, nodes in zip(model.arch_sampler.params, model.arch_sampler.group_var_names):
                assert params.size(0) == len(nodes)
                # print(params, nodes)
                for p, n in zip(params.unbind(), nodes):
                    node_grads[n] = p
            # print(arch_grad)
            # print(grad_dict)
        else:
            ent_mean = None
            ent_sampling = None
            grad_dict = None
            proba_dict = None
            node_grads = None
        # selected_optims = optimizer

        # if model.cur_split is not None and len(optimizer) > 1:
        #     print(type(optimizer))
        #     opt = optimizer[model.cur_split]
        #     pp = opt.param_groups[0]['params'][0].grad
        #     pp_grad = (pp**2).mean() if pp is not None else pp
        #     print('Step {} on grads {}'.format(model.cur_split, pp_grad))
        #     if pp_grad.item() < 0.0003:
        # opt.step()
        # else:
        additional_info = {'arch_entropy_avg': ent_mean,
                           'arch_entropy_sample': ent_sampling,
                           'arch_grads': grad_dict,
                           'node_grads': node_grads,
                           'arch_probas': proba_dict,
                           **details}
        # print(additional_info)
        return output_transform(x, y, y_pred, loss, additional_info)

    return Engine(_update)


def create_supervised_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=
                                lambda x, y, y_pred: (y_pred, y,)):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # z is optional (e.g. task ids)
            x, y, *z = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(*(x, *z))
            # if hasattr(model, 'arch_sampler'):
            #     ent = model.arch_sampler.entropy().mean()

            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

