import torch

from Value import Value
from Visualize import *
from RandomNetwork.MLP import MLP


def lol():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f
    L.label = 'L'
    x = draw_dot(L)
    x.render('test/digraph.gv', view=True)


def repeat():
    a = Value(3.0, label='a')
    b = a + a
    b.label = 'b'
    b.backward()


def simple_example():
    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')
    d = a + b
    d.label = 'd'
    e = a + b
    e.label = 'e'
    f = d + e
    f.label = 'f'


def example():
    # inputs x1, x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights w1, w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of the neuron
    b = Value(6.8813735870195432, label='b')
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = 'x1w1'
    x2w2 = x2 * w2
    x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b
    n.label = 'n'
    e = (2 * n).exp()
    o = (e - 1) / (e + 1)
    # o = n.tanh()
    # o.label = 'o'
    o.backward()
    x = draw_dot(o)
    x.render('test/digraph.gv', view=True)


def pytorch_example():
    x1 = torch.Tensor([2.0]).double()
    x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()
    x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()
    w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double()
    b.requires_grad = True
    n = x1 * w1 + x2 * w2 + b
    o = torch.tanh(n)
    print(o.data.item())
    o.backward()

    print('---')
    print('x2', x2.grad.item())
    print('w2', w2.grad.item())
    print('x1', x1.grad.item())
    print('w1', w1.grad.item())


def main():
    # x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    # z = draw_dot(n(x))
    # z.render('test/digraph.gv', view=True)

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, 0.5],
    ]
    ys = [1.0, -1.0, -1.0, -1.0]

    for k in range(20):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(k, loss.data)

    print(ypred)
    z = draw_dot(loss)
    z.render('test/digraph.gv', view=True)


if __name__ == '__main__':
    main()
