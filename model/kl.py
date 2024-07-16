import torch
import torch.nn.functional as F
import random
class KLLoss(torch.nn.Module):

    def __init__(self, args, device):
        super(KLLoss, self).__init__()
        self.eval_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.memory_size = args.memory_size
        self.sample_size = args.sample_size
        self.device = device
        self.memory = torch.randn_like(torch.zeros(self.memory_size, dtype=torch.float32, requires_grad=False)).to(device)

    def forward(self, logits):

        index = torch.LongTensor(random.sample(range(self.memory_size), logits.shape[0])).to(self.device)

        ref = torch.index_select(self.memory, 0, index)


        logp_x = torch.normal(ref, torch.var(ref))
        logp_x = torch.sigmoid(logp_x)

        edge_y = torch.normal(logits, torch.var(logits))
        edge_y = torch.sigmoid(edge_y)

        p_x = F.log_softmax(logp_x, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)

        edge_loss = self.eval_loss(p_x, p_y) * 100

        #
        # prior_mean = torch.mean(logits)
        # prior_variance = torch.var(logits)
        #
        #
        # posterior_mean = logits
        # posterior_variance = torch.var(logits - labels)
        #
        #
        # kl_divergence = 0.5 * (torch.log(prior_variance) - torch.log(posterior_variance) +
        #                        (posterior_variance + (posterior_mean - prior_mean) ** 2) / prior_variance - 1)
        # kl_loss = kl_divergence.mean()

        return edge_loss
        # return kl_loss


