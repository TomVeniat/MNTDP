from sacred.randomness import get_seed

from ctrl.instances.image_dataset_tree import ImageDatasetTree
from ctrl.instances.md_tree import MultiDomainDatasetTree
from ctrl.strategies import InputDomainMutationStrategy, SplitStrategy, \
    IncrementalStrategy, RandomMutationStrategy, DataStrategy, \
    AttributeStrategy, MixedStrategy, LabelPermutationStrategy
from ctrl.tasks.task_generator import TaskGenerator
from ctrl.transformations import RandomNNTransformationTree, \
    ImgRotationTransformationTree, \
    RandomPermutationsTransformation, IdentityTransformation, \
    NoisyNNTransformationTree, RainbowTransformationTree


def get_dataset_by_name(name):
    if name in ['cifar10_tree', 'cifar100_tree', 'mnist_tree', 'svhn_tree',
                'fashion_mnist_tree', 'dtd_tree', 'aircraft_tree']:
        return ImageDatasetTree
    if name.startswith('md_tree'):
        return MultiDomainDatasetTree

    if name == 'nn_x_transformation':
        return RandomNNTransformationTree
    if name == 'img_rot_x_transformation':
        return ImgRotationTransformationTree
    if name == 'randperm_x_transformation':
        return RandomPermutationsTransformation
    if name == 'id_x_transformation':
        return IdentityTransformation
    if name == 'noisy_nn_x_transformation':
        return NoisyNNTransformationTree
    if name == 'rainbow_x_transformation':
        return RainbowTransformationTree

    if name == 'transfo':
        return InputDomainMutationStrategy
    if name == 'split':
        return SplitStrategy
    if name == 'incremental':
        return IncrementalStrategy
    if name == 'random':
        return RandomMutationStrategy
    if name == 'data':
        return DataStrategy
    if name == 'attributes':
        return AttributeStrategy
    if name.startswith('mixed'):
        return MixedStrategy
    if name == 'label_permut':
        return LabelPermutationStrategy

    if name == 'task_gen':
        return TaskGenerator

    raise NotImplementedError(name)


def init_dataset(_rnd, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, dict):
                v = init_dataset(_rnd=_rnd, **v)
        kwargs[k] = v
    if '_name' in kwargs:
        return get_dataset_by_name(kwargs.pop('_name'))(seed=get_seed(_rnd), **kwargs)
    else:
        return kwargs