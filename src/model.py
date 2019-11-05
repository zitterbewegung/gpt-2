import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os

from tensorflow.contrib.cluster_resolver import TPUClusterResolver

def get_tpu_addr():
    # Get the TPU's location
    if 'COLAB_TPU_ADDR' not in os.environ:
        return None
    else:
        return TPUClusterResolver().get_master()

def get_tpu_shards():
    if 'SHARDS' not in os.environ:
        return 8
    else:
        return int(os.environ['SHARDS'])

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        tpu_address=get_tpu_addr(),
        shards=get_tpu_shards()
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02, b_init=0):
    with tf.variable_scope(scope):
        shape = shape_list(x)
        *start, nx = shape
        w = conv1d_w(nf, nx, w_init_stdev=w_init_stdev)
        b = conv1d_b(nf, b_init=b_init)
        c = conv1d_op(x, w, b, nf, shape=shape)
        return c

def conv1d_w(nf, nx, *, w_init_stdev=0.02):
    return tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))

def conv1d_b(nf, *, b_init=0):
    return tf.get_variable('b', [nf], initializer=tf.constant_initializer(b_init))

def conv1d_op(x, w, b, nf, shape=None):
    if shape is None:
        shape = shape_list(x)
    *start, nx = shape or shape_list(x)
    X = tf.reshape(x, [-1, nx])
    W = tf.reshape(w, [-1, nf])
    Y = tf.matmul(X, W) + b
    return tf.reshape(Y, start+[nf])

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


# http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
import itertools
def group(n, iterable, fillvalue=None):
    "group(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

_tpus = None
def get_tpus(hparams):
    if hparams.shards == 1 or hparams.shards == 8:
        return None
    global _tpus
    if _tpus is None and hparams.tpu_address is not None:
        with tf.Session(hparams.tpu_address) as sess:
            devices = sess.list_devices()
            tpus = devices[-1 - 8:-1]
        _tpus = list(group(hparams.shards, tpus))
    return _tpus

def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        ln_1 = norm(x, 'ln_1')
        a, present = attn(ln_1, 'attn', nx, past=past, hparams=hparams)
        x = x + a
        ln_2 = norm(x, 'ln_2')
        if False:
            def op(input):
                # input = tf.transpose(input)
                shards = nx // input.shape[-1]
                n_state = nx * 4 // shards
                if 'GPT2_N_STATE' in os.environ:
                    n_state = int(os.environ['GPT2_N_STATE'])
                if 'GPT2_DEBUG' in os.environ:
                    print('shards', shards, x.shape, input.shape, nx, n_state)
                return mlp(input, 'mlp', n_state, hparams=hparams)
            if hparams.tpu_address is not None:
                m = tf.contrib.tpu.shard(op, [ln_2], input_shard_axes=[2], output_shard_axes=[2], num_shards=hparams.shards, device_assignment=get_tpus(hparams))
            else:
                m = op(ln_2)
            #m = tf.reshape(m, x.shape)
        else:
            n_state = nx*4
            m = mlp(ln_2, 'mlp', n_state, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        def op(h_flat, wte):
            result = tf.matmul(h_flat, wte, transpose_b=True)
            if 'GPT2_DEBUG' in os.environ:
                print('op', h_flat, wte, result)
            return result
        if hparams.tpu_address is not None:
            input_shard_axis_0 = 1 if not 'GPT2_INPUT_SHARD_AXIS_0' in os.environ else int(os.environ['GPT2_INPUT_SHARD_AXIS_0'])
            input_shard_axis_1 = 1 if not 'GPT2_INPUT_SHARD_AXIS_1' in os.environ else int(os.environ['GPT2_INPUT_SHARD_AXIS_1'])
            output_shard_axis_0 = 1 if not 'GPT2_OUTPUT_SHARD_AXIS_0' in os.environ else int(os.environ['GPT2_OUTPUT_SHARD_AXIS_0'])
            output_reduce_axis = 0 if not 'GPT2_OUTPUT_REDUCE_AXIS' in os.environ else int(os.environ['GPT2_OUTPUT_REDUCE_AXIS'])
            logits0 = tf.contrib.tpu.shard(op, [h_flat, wte], input_shard_axes=[input_shard_axis_0, input_shard_axis_1], output_shard_axes=[output_shard_axis_0], num_shards=hparams.shards, device_assignment=get_tpus(hparams))
            if output_reduce_axis >= 0:
                logits0 = tf.reduce_sum(logits0, axis=output_reduce_axis, keepdims=True)
        else:
            logits0 = op(h_flat, wte)
        logits = tf.reshape(logits0, [batch, sequence, hparams.n_vocab])
        if 'GPT2_DEBUG' in os.environ:
            print('logits', logits0, logits, batch, sequence, hparams.n_vocab)
        results['logits'] = logits
        return results
