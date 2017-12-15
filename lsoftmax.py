import math

import torch
from torch import nn
from torch.autograd import Variable

# import torch
from scipy.special import binom
# from torch.autograd import Function


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))

        self.divisor = math.pi / self.margin
        self.coeffs = binom(margin, range(0, margin + 1, 2))
        self.cos_exps = range(self.margin, -1, -2)
        self.sin_sq_exps = range(len(self.cos_exps))
        self.signs = [1]
        for i in range(1, len(self.sin_sq_exps)):
            self.signs.append(self.signs[-1] * -1)

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight.data.t())

    def find_k(self, cos):
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            logit = input.matmul(self.weight)
            batch_size = logit.size(0)
            logit_target = logit[range(batch_size), target]
            weight_target_norm = self.weight[:, target].norm(p=2, dim=0)
            input_norm = input.norm(p=2, dim=1)
            # norm_target_prod: (batch_size,)
            norm_target_prod = weight_target_norm * input_norm
            # cos_target: (batch_size,)
            cos_target = logit_target / (norm_target_prod + 1e-10)
            sin_sq_target = 1 - cos_target**2

            num_ns = self.margin//2 + 1
            # coeffs, cos_powers, sin_sq_powers, signs: (num_ns,)
            coeffs = Variable(input.data.new(self.coeffs))
            cos_exps = Variable(input.data.new(self.cos_exps))
            sin_sq_exps = Variable(input.data.new(self.sin_sq_exps))
            signs = Variable(input.data.new(self.signs))

            cos_terms = cos_target.unsqueeze(1) ** cos_exps.unsqueeze(0)
            sin_sq_terms = (sin_sq_target.unsqueeze(1)
                            ** sin_sq_exps.unsqueeze(0))

            cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
                          * cos_terms * sin_sq_terms)
            cosm = cosm_terms.sum(1)
            k = self.find_k(cos_target)

            ls_target = norm_target_prod * (((-1)**k * cosm) - 2*k)
            logit[range(batch_size), target] = ls_target
            return logit
        else:
            assert target is None
            return input.matmul(self.weight)


'''
class LSoftmaxLinearOp(Function):

    @staticmethod
    def forward(ctx, weight, input, target, margin):
        """
        Args:
            weight: A float tensor of size (dim, num_classes).
            input: A float tensor of size (batch_size, dim).
            target: A float tensor of size (batch_size,).
            margin: An integer specifying the margin.

        Returns:
            logit: A tensor of size (batch_size, num_classes)
                containing large-margin unnormalized probabilities.
        """

        # w_norm: (1, num_classes)
        w_norm = weight.norm(p=2, dim=0, keepdim=True)
        # x_norm: (batch_size, 1)
        x_norm = input.norm(p=2, dim=1, keepdim=True)
        # wx_norm_prod: (batch_size, num_classes)
        wx_norm_prod = w_norm * x_norm
        # logit: (batch_size, num_classes)
        logit = input.matmul(weight)

        batch_size = logit.size(0)
        # xxx_target: (batch_size,)
        wx_dot_target = logit[range(batch_size), target]
        wx_norm_prod_target = wx_norm_prod[range(batch_size), target]
        cos_target = wx_dot_target / wx_norm_prod_target

        num_ns = margin//2 + 1
        constants = cos_target.new(comb(margin, range(0, margin + 1, 2)))
        if margin % 2:
            cos_powers = cos_target.new(num_ns, batch_size).fill_(1)
        else:
            cos_powers = cos_target.repeat(num_ns, 1)
        sin_powers = cos_target.new(num_ns, batch_size).fill_(1)
        signs = cos_target.new(num_ns).fill_(1)
        cos_target_sq = cos_target * cos_target
        sin_target_sq = 1 - cos_target_sq
        for n in range(1, margin//2 + 1):
            cos_powers[-n - 1] = cos_powers[-n] * cos_target_sq
            sin_powers[n] = sin_powers[n - 1] * sin_target_sq
            signs = -signs[n - 1]
        cosm_target = (signs.unsqueeze(1) * constants.unsqueeze(1)
                      * cos_powers * sin_powers).sum(0)

        # Let's find k.
        angle_target = cos_target.acos()
        divisor = math.pi / margin
        k = (angle_target / divisor).floor_()

        logit_target = (torch.pow(-1, k) * wx_norm_prod_target
                        * (cosm_target - 2*k))
        logit[range(batch_size), target] = logit_target

        ctx.save_for_backward(

        )
        return logit

    @staticmethod
    def backward(ctx, grad_output):
        weight, input, target = ctx.saved_variables
        grad_weight, grad_input = None
        k = ctx.k

        if ctx.needs_input_grad[0]:
'''
