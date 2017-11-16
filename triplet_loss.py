import mxnet as mx

# [ fa fp fn, fa fp fn, ..., fa fp fn ]
# bs x 3 x feat_dim
def triplet_loss(data):
    f = mx.sym.split(data=data, num_outputs=3, axis=1, squeeze_axis=True)
    fa = f[0]
    fp = f[1]
    fn = f[2]

    one = mx.symbol.Variable("one", init=mx.initializer.One(), attr={'lr_mult': '0.0'}, shape=(kwargs['batch_size'],))
    fs = fa - fp
    fd = fa - fn
    fs = fs * fs
    fd = fd * fd
    fs = mx.sym.sum(fs, axis=1, keepdims=0)
    fd = mx.sym.sum(fd, axis=1, keepdims=0)
    loss = fd - fs
    loss = one - loss
    loss = mx.symbol.Activation(data=loss, act_type='relu')
    loss = mx.symbol.MakeLoss(loss)
    return loss
