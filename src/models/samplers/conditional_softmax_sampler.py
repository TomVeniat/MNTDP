import networkx as nx
import torch
import torch.nn.functional as f
from torch.distributions import Categorical

from src.models.samplers.softmax_sampler import SoftmaxArchGenerator


class CondiSoftmaxArchGenerator(SoftmaxArchGenerator):
    def sample_archs(self, batch_size, probas, force_deterministic=False):
        """
        Hook called by pytorch before each forward
        :param _: Current module
        :param input: Input given to the module's forward
        :return:
        """
        torch.autograd.set_detect_anomaly(True)
        deterministic = not self.training and self.deter_eval \
                        or force_deterministic
        self._check_probas(probas, self.all_same)

        # Check the compatibility with the batch_size
        if probas.size(0) != batch_size:
            if probas.size(0) != 1:
                raise ValueError('Sampling probabilities dimensions {} '
                                 'doesn\'t match with batch size {}.'
                                 .format(probas.size(), batch_size))
            # if not self.all_same:
            #     probas = probas.expand(batch_size, -1)

        samplings = torch.zeros_like(probas)
        entropies = []
        log_probs = []
        if not self.all_same:
            samplings = samplings.repeat(batch_size, 1)
        order = list(nx.topological_sort(self.graph))

        has_input_mat = torch.zeros(len(order), batch_size).bool().to(probas.device)
        has_input = {n: t for n, t in zip(order, has_input_mat.unbind(0))}
        has_input_mat[0] = True
        # has_input[order[0]].fill_(True)
        # max_group_size = max(self.group_sizes)
        for node in order:
            node_input = has_input[node]
            if not node_input.any():
                continue

            grp_id = self.group_ids_index.get(node)
            if grp_id is None:
                next = self.graph.successors(node)
                for neighbor in next:
                    # has_input[neighbor] |= node_input
                    has_input[neighbor] = has_input[neighbor] | node_input
                # has_input.update(next)
                # raise NotImplementedError()
            else:
                # if not node_input.all() and not deterministic:
                #     print('\b')
                mask = self.group_masks[grp_id]
                masked_probas = probas[mask.unsqueeze(0)].unsqueeze(0)
                if not self.all_same:
                    masked_probas = masked_probas.repeat(node_input.sum(), 1)
                    # masked_probas = masked_probas[node_input]
                distrib = Categorical(masked_probas)

                assert masked_probas.dim() == 2
                if deterministic or masked_probas.size(1) == 1:
                    index = masked_probas.argmax(1)
                else:
                    index = distrib.sample()

                samplable_nodes = self.group_var_names[grp_id]
                assert masked_probas.size(1) == len(samplable_nodes)
                for idx, next_node in enumerate(samplable_nodes):
                    has_input[next_node][node_input] = (has_input[next_node][node_input] | (index == idx))
                    # has_input[next_node][node_input] |= (index == idx)
                    next = list(self.graph.successors(next_node))
                    assert len(next) == 1
                    # Need to verifiy that this node can't have any other input
                    assert len(list(self.graph.predecessors(next_node))) == 1
                    # has_input[next[0]] |= has_input[next_node]

                new_sampling = f.one_hot(index, masked_probas.size(-1)).float()
                # new_sampling[~node_input] = 0

                expanded_mask = mask.repeat(samplings.size(0), 1)
                expanded_mask[~node_input] = False
                samplings[expanded_mask] = new_sampling.flatten()

                ent = torch.zeros(samplings.size(0)).to(probas.device)
                ent[node_input] = distrib.entropy()
                entropies.append(ent)

                lp = torch.zeros(samplings.size(0)).to(probas.device)
                lp = lp.masked_scatter(node_input.clone(), distrib.log_prob(index))
                # lp[node_input] = distrib.log_prob(index)
                log_probs.append(lp)


        if self.all_same:
            samplings = samplings.expand(batch_size, -1)

        # self._seq_probas.append(probas)
        if entropies:
            self.distrib_entropies.append(torch.stack(entropies, 1))
        if log_probs:
            self.log_probas.append(torch.stack(log_probs, 1))
        return samplings
