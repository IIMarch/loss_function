import mxnet as mx

def huber_loss(predictions, labels, delta):
    error = predictions - labels
    abs_error = mx.sym.abs(error)
    quadratic = mx.sym.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic * quadratic + delta * linear
    return losses
