import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/0.11.0/python')
import mxnet as mx

def weighted_softmax_loss(data, weighted, label):
    # bs x num_classes
    logits = mx.sym.softmax(data=data, name='logits')

    # bs
    probs = mx.sym.pick(data=logits, index=label, axis=1, name='probs')

    # input: bs x num_classes -> weighted
    weighted = mx.sym.pick(data=weighted, index=label, axis=1, name='weighted_cls')

    log_probs = - mx.sym.broadcast_mul(weighted, mx.sym.log(data=probs, name='log_probs'))
    loss = mx.sym.sum(data=log_probs, name='loss')
    loss = mx.sym.MakeLoss(loss)

    return loss

