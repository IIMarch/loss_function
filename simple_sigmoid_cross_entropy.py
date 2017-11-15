import mxnet as mx

def simple_sigmoid_cross_entropy(pred, target, num_classes):
    clss = mx.symbol.Activation(data=pred, act_type='sigmoid', name='sigmoid')
    clss = mx.symbol.split(data=clss, num_outputs=num_classes, axis=1)
    labels = mx.symbol.split(data=target, num_outputs=num_classes, axis=1)

    losses = []
    for i in range(num_classes):
        losses.append(
          mx.symbol.MakeLoss(
            -mx.symbol.broadcast_add(
              mx.symbol.broadcast_mul(labels[i], mx.symbol.log(clss[i])),
              mx.symbol.broadcast_mul((1-labels[i]), mx.symbol.log(1-clss[i]))
            )
          )
        )
    train = mx.symbol.Group(losses)
    return train
