import unittest

import numpy as np
import torch
from torch.autograd import Variable

from lsoftmax import LSoftmaxLinear


class LSoftmaxTestCase(unittest.TestCase):

    def test_trivial(self):
        m = LSoftmaxLinear(input_dim=2, output_dim=4, margin=1)
        x = Variable(torch.arange(6).view(3, 2))
        target = Variable(torch.LongTensor([3, 0, 1]))

        m.eval()
        eval_out = m.forward(input=x).data.tolist()
        m.train()
        train_out = m.forward(input=x, target=target).data.tolist()
        np.testing.assert_array_almost_equal(eval_out, train_out)

    def test_odd(self):
        m = LSoftmaxLinear(input_dim=2, output_dim=4, margin=3)
        m.weight.data.copy_(torch.arange(-4, 4).view(2, 4))
        x = Variable(torch.arange(6).view(3, 2))
        target = Variable(torch.LongTensor([3, 0, 1]))

        m.eval()
        eval_out = m.forward(input=x).data.tolist()
        eval_gold = [[0, 1, 2, 3], [-8, -3, 2, 7], [-16, -7, 2, 11]]
        np.testing.assert_array_almost_equal(eval_out, eval_gold, decimal=5)

        m.train()
        train_out = m.forward(input=x, target=target).data.tolist()
        train_gold = [[0, 1, 2, 1.7999999999999996],
                      [-43.53497425357768, -3, 2, 7],
                      [-16, -58.150571999218542, 2, 11]]
        np.testing.assert_array_almost_equal(train_out, train_gold, decimal=5)

    def test_even(self):
        m = LSoftmaxLinear(input_dim=2, output_dim=4, margin=4)
        m.weight.data.copy_(torch.arange(-4, 4).view(2, 4))
        x = Variable(torch.arange(6).view(3, 2))
        target = Variable(torch.LongTensor([3, 0, 1]))

        m.eval()
        eval_out = m.forward(input=x).data.tolist()
        eval_gold = [[0, 1, 2, 3], [-8, -3, 2, 7], [-16, -7, 2, 11]]
        np.testing.assert_array_almost_equal(eval_out, eval_gold, decimal=5)

        m.train()
        train_out = m.forward(input=x, target=target).data.tolist()
        train_gold = [[0, 1, 2, 0.88543774484714499],
                      [-67.844100922931872, -3, 2, 7],
                      [-16, -77.791173935544478, 2, 11]]
        np.testing.assert_array_almost_equal(train_out, train_gold, decimal=5)


if __name__ == '__main__':
    unittest.main()
