import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/0.11.0/python')
import mxnet as mx
import numpy as np

def weighted_softmax_loss(data, weighted, label):
    # bs x num_classes
    logits = mx.sym.softmax(data=data, name='logits')

    # bs
    probs = mx.sym.pick(data=logits, index=label, axis=1, name='probs')
    log_probs = - mx.sym.broadcast_mul(weighted, mx.sym.log(data=probs, name='log_probs'))
    loss = mx.sym.sum(data=log_probs, name='loss')
    loss = mx.sym.MakeLoss(loss)

    return loss


if __name__ == '__main__':
    data = mx.sym.Variable(name='data')
    weighted = mx.sym.Variable(name='weighted')
    label = mx.sym.Variable(name='label')

    loss = weighted_softmax_loss(data, weighted, label)
    #loss = mx.sym.SoftmaxOutput(data=data, label=label, name='loss')
    data_shapes = {'data':(1, 4), 'weighted':(4,), 'label':(1,)}
    #data_shapes = {'data':(1, 4), 'label':(1,)}

    exe_ = loss.simple_bind(ctx=mx.cpu(), **data_shapes)

    w = np.array([0.25,0.25,0.25,0.25])
    label = np.array([2])
    data = np.array([[1,2,3,4]])

    mx.nd.array(w).copyto(exe_.arg_dict['weighted'])
    mx.nd.array(label).copyto(exe_.arg_dict['label'])
    mx.nd.array(data).copyto(exe_.arg_dict['data'])

    exe_.forward(is_train=True)
    exe_.backward()

    outputs = [output.asnumpy() for output in exe_._get_outputs()]
    print exe_.grad_dict

