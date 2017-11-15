import mxnet as mx

def stable_sigmoid_cross_entropy(pred, target):
    pred_ge_zero = pred.__ge__(0)
    a_term = mx.symbol.broadcast_mul(pred,
        mx.symbol.broadcast_sub(target, pred_ge_zero))
    b_term = mx.symbol.log(
        1. + mx.symbol.exp(mx.symbol.broadcast_sub(pred,
            2. * mx.symbol.broadcast_mul(pred, pred_ge_zero)
            )
        )
    )
    loss = -mx.symbol.mean(
        mx.symbol.sum_axis(mx.symbol.broadcast_sub(a_term, b_term), axis=1)
    )
    return mx.symbol.MakeLoss(loss)
