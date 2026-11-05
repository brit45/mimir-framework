#!/usr/bin/env python3
"""Fixed-input, forward-only architecture benchmark adapters."""
import json
import math
import os
from pathlib import Path
import sys
import statistics
import time
import traceback


def _validate_config(config):
    required = ('benchmark_kind', 'case', 'architecture', 'model_config',
                'input_kind', 'input_size', 'token_count', 'vocab_size',
                'expected_output_size', 'warmup', 'iterations', 'mimir')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError('Missing architecture config fields: ' + ', '.join(missing))
    if config['benchmark_kind'] != 'architecture':
        raise ValueError("benchmark_kind must be 'architecture'")
    if config['input_kind'] not in ('float', 'tokens'):
        raise ValueError("input_kind must be 'float' or 'tokens'")
    if config['input_kind'] == 'float' and config['input_size'] <= 0:
        raise ValueError('input_size must be positive for float inputs')
    if config['input_kind'] == 'tokens' and (config['token_count'] <= 0 or config['vocab_size'] <= 0):
        raise ValueError('token_count and vocab_size must be positive for token inputs')
    if config['expected_output_size'] <= 0 or config['warmup'] < 0 or config['iterations'] <= 0:
        raise ValueError('Invalid output size or iteration count')
    if not isinstance(config['model_config'], dict) or not isinstance(config['mimir'], dict):
        raise ValueError('model_config and mimir must be objects')


def _seed(framework, backend, seed):
    if framework == 'pytorch':
        import torch
        torch.manual_seed(seed)
        if backend == 'cuda':
            torch.cuda.manual_seed_all(seed)
    elif framework == 'tensorflow':
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    elif framework == 'jax':
        import numpy as np
        np.random.seed(seed)


def _torch_model(framework, config):
    import torch
    from torch import nn
    family = config['family']
    c = config['model_config']

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            if family == 'basic_mlp':
                dims = [c['input_dim']] + [c['hidden_dim']] * c['hidden_layers'] + [c['output_dim']]
                self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
            elif family == 'vgg16':
                channels = c['base_channels']
                blocks = (2, 2, 3, 3, 3)
                self.convs = nn.ModuleList()
                current = c['image_c']
                for block, count in enumerate(blocks):
                    output = channels * (2 ** block)
                    for _ in range(count):
                        self.convs.append(nn.Conv2d(current, output, 3, padding=1, bias=False))
                        current = output
                    self.convs.append(nn.Conv2d(current, current, 3, stride=2, padding=1, bias=False))
                self.fc1 = nn.Linear(current, c['fc_hidden'])
                self.fc2 = nn.Linear(c['fc_hidden'], c['fc_hidden'])
                self.head = nn.Linear(c['fc_hidden'], c['num_classes'])
            elif family == 'resnet':
                self.stem = nn.Conv2d(c['image_c'], c['base_channels'], 3, stride=2, padding=1, bias=False)
                self.blocks = nn.ModuleList()
                current = c['base_channels']
                for stage, count in enumerate((c['blocks1'], c['blocks2'], c['blocks3'], c['blocks4'])):
                    output = c['base_channels'] * (2 ** stage)
                    for index in range(count):
                        stride = 2 if stage and index == 0 else 1
                        self.blocks.append(nn.ModuleList([
                            nn.Conv2d(current, output, 3, stride=stride, padding=1, bias=False),
                            nn.Conv2d(output, output, 3, padding=1, bias=False),
                            nn.Conv2d(current, output, 1, stride=stride, bias=False)
                            if stride != 1 or current != output else nn.Identity(),
                        ]))
                        current = output
                self.head = nn.Linear(current, c['num_classes'])
            elif family == 'vae_conv':
                base, latent = c['base_channels'], c['latent_c']
                self.enc1 = nn.Conv2d(c['image_c'], base, 3, stride=2, padding=1)
                self.enc2 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
                self.mu = nn.Conv2d(base * 2, latent, 1)
                self.logvar = nn.Conv2d(base * 2, latent, 1)
                self.dec1 = nn.Conv2d(latent, base * 2, 3, padding=1)
                self.dec2 = nn.Conv2d(base * 2, base, 3, padding=1)
                self.dec3 = nn.Conv2d(base, c['image_c'], 3, padding=1)
            elif family in ('transformer', 'causal_transformer', 'vit'):
                d, heads = c['d_model'], c['num_heads']
                self.embed = nn.Embedding(c['vocab_size'], d) if family != 'vit' else None
                self.ln1 = nn.LayerNorm(d)
                self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
                self.ln2 = nn.LayerNorm(d)
                if family == 'vit':
                    self.ff1 = nn.Linear(d, c['mlp_hidden'])
                    self.ff2 = nn.Linear(c['mlp_hidden'], d)
                    self.head = nn.Linear(d, c['output_dim'])
                else:
                    size = c['seq_len'] * d
                    self.ff1 = nn.Linear(size, c['mlp_hidden'])
                    self.ff2 = nn.Linear(c['mlp_hidden'], size)
                    self.head = nn.Linear(size, c['output_dim'])
            elif family == 'unet':
                self.downs = nn.ModuleList()
                current = c['image_c']
                for depth in range(c['depth']):
                    channels = c['base_channels'] * (2 ** depth)
                    self.downs.append(nn.ModuleList([
                        nn.Conv2d(current, channels, 3, padding=1),
                        nn.Conv2d(channels, channels, 3, padding=1),
                        nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                    ]))
                    current = channels
                self.bottleneck = nn.Conv2d(current, current * 2, 3, padding=1)
                current *= 2
                self.ups = nn.ModuleList()
                for depth in reversed(range(c['depth'])):
                    channels = c['base_channels'] * (2 ** depth)
                    self.ups.append(nn.ModuleList([
                        nn.Conv2d(current + channels, channels, 3, padding=1),
                        nn.Conv2d(channels, channels, 3, padding=1),
                    ]))
                    current = channels
                self.out = nn.Conv2d(current, c['image_c'], 1)
            else:
                raise ValueError('Unknown architecture family: ' + family)

        def forward(self, value):
            if family == 'basic_mlp':
                for layer in self.layers[:-1]:
                    value = torch.nn.functional.gelu(layer(value))
                return self.layers[-1](value)
            if family == 'vgg16':
                for conv in self.convs:
                    value = conv(value)
                    value = torch.nn.functional.layer_norm(value, value.shape[1:])
                    value = torch.relu(value)
                value = value.mean(dim=(-2, -1))
                value = torch.nn.functional.gelu(self.fc1(value))
                return self.head(torch.nn.functional.gelu(self.fc2(value)))
            if family == 'resnet':
                value = torch.relu(self.stem(value))
                for conv1, conv2, skip in self.blocks:
                    residual = skip(value)
                    value = torch.relu(conv1(value))
                    value = torch.relu(conv2(value) + residual)
                return self.head(value.mean(dim=(-2, -1)))
            if family == 'vae_conv':
                value = torch.nn.functional.silu(self.enc1(value))
                value = torch.nn.functional.silu(self.enc2(value))
                mu, logvar = self.mu(value), self.logvar(value)
                value = mu
                value = torch.nn.functional.silu(self.dec1(value))
                value = torch.nn.functional.interpolate(value, scale_factor=2, mode='nearest')
                value = torch.nn.functional.silu(self.dec2(value))
                value = torch.nn.functional.interpolate(value, scale_factor=2, mode='nearest')
                recon = torch.tanh(self.dec3(value))
                return recon.flatten()
            if family in ('transformer', 'causal_transformer'):
                value = self.embed(value)
                x = self.ln1(value)
                mask = torch.triu(torch.ones(c['seq_len'], c['seq_len'], device=x.device, dtype=torch.bool), diagonal=1) if c['causal'] else None
                attended, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
                value = value + attended
                flattened = value.flatten(1)
                residual = flattened
                x = self.ln2(value).flatten(1)
                value = self.ff2(torch.nn.functional.gelu(self.ff1(x))) + residual
                return self.head(value)
            if family == 'vit':
                x = self.ln1(value)
                attended, _ = self.attn(x, x, x, need_weights=False)
                value = value + attended
                x = self.ln2(value)
                value = value + self.ff2(torch.nn.functional.gelu(self.ff1(x)))
                return self.head(value.mean(dim=1))
            skips = []
            for conv1, conv2, down in self.downs:
                value = torch.relu(conv1(value))
                value = torch.relu(conv2(value))
                skips.append(value)
                value = torch.relu(down(value))
            value = torch.relu(self.bottleneck(value))
            for layers, skip in zip(self.ups, reversed(skips)):
                value = torch.nn.functional.interpolate(value, size=skip.shape[-2:], mode='nearest')
                value = torch.cat((value, skip), dim=1)
                value = torch.relu(layers[0](value))
                value = torch.relu(layers[1](value))
            return self.out(value).flatten()

    model = Network()
    model.eval()
    return model


def _input_data(config, framework, backend):
    import numpy as np
    c = config['model_config']
    if config['input_kind'] == 'tokens':
        tokens = np.asarray([(i * 7) % config['vocab_size'] for i in range(config['token_count'])], dtype=np.int64)
        if framework == 'pytorch':
            import torch
            return torch.as_tensor(tokens, dtype=torch.long, device=backend_device(framework, backend))
        if framework == 'tensorflow':
            import tensorflow as tf
            return tf.convert_to_tensor(tokens[None, :], dtype=tf.int32)
        import jax.numpy as jnp
        return jnp.asarray(tokens[None, :], dtype=jnp.int32)
    values = np.asarray([(i % 31 - 15) / 15.0 for i in range(config['input_size'])], dtype=np.float32)
    if config['family'] in ('vgg16', 'resnet', 'vae_conv', 'unet'):
        values = values.reshape(1, c['image_c'], c['image_h'], c['image_w'])
        if framework in ('tensorflow', 'jax'):
            values = values.transpose(0, 2, 3, 1)
    elif config['family'] == 'vit':
        values = values.reshape(1, c['num_tokens'], c['d_model'])
    else:
        values = values.reshape(1, c['input_dim'])
    if framework == 'pytorch':
        import torch
        return torch.as_tensor(values, dtype=torch.float32, device=backend_device(framework, backend))
    if framework == 'tensorflow':
        import tensorflow as tf
        return tf.convert_to_tensor(values, dtype=tf.float32)
    import jax.numpy as jnp
    return jnp.asarray(values, dtype=jnp.float32)


def backend_device(framework, backend):
    if framework == 'pytorch':
        import torch
        return {'cpu': torch.device('cpu'), 'cuda': torch.device('cuda:0'),
                'tpu': torch.device('xla:0')}[backend]
    raise ValueError('backend_device is only used for PyTorch')


def _run_pytorch(config, backend):
    if backend == 'tpu':
        os.environ['PJRT_DEVICE'] = 'TPU'
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if backend == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
    xm = None
    if backend == 'tpu':
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        if xr.device_type() != 'TPU':
            raise RuntimeError('PyTorch/XLA did not prove TPU placement')
        device = xm.xla_device()
    else:
        device = backend_device('pytorch', backend)
    model = _torch_model('pytorch', config).to(device)
    if any(p.device.type != device.type for p in model.parameters()):
        raise RuntimeError('PyTorch parameters are not on the requested device')
    inputs = _input_data(config, 'pytorch', backend)
    if inputs.device.type != device.type:
        raise RuntimeError('PyTorch input is not on the requested device')
    def sync(value):
        if backend == 'cuda':
            torch.cuda.synchronize(device)
        elif xm is not None:
            xm.mark_step()
            xm.wait_device_ops()
        return value
    def forward():
        with torch.inference_mode():
            return model(inputs)
    return model, inputs, forward, sync, lambda out: out.device.type == device.type, str(device)


def _tf_model(tf, config):
    from tensorflow import keras
    family, c = config['family'], config['model_config']
    L = keras.layers
    inputs = keras.Input(shape=((config['token_count'],) if config['input_kind'] == 'tokens' else
                                (c['image_h'], c['image_w'], c['image_c']) if family in ('vgg16', 'resnet', 'vae_conv', 'unet') else
                                (c['num_tokens'], c['d_model']) if family == 'vit' else (c['input_dim'],)),
                         dtype='int32' if config['input_kind'] == 'tokens' else 'float32')
    if family == 'basic_mlp':
        x = inputs
        for _ in range(c['hidden_layers']):
            x = L.Dense(c['hidden_dim'], activation='gelu')(x)
        outputs = L.Dense(c['output_dim'])(x)
    elif family in ('transformer', 'causal_transformer', 'vit'):
        if config['input_kind'] == 'tokens':
            x = L.Embedding(c['vocab_size'], c['d_model'])(inputs)
        else:
            x = inputs
        causal = c.get('causal', False)
        attn = L.MultiHeadAttention(c['num_heads'], c['d_model'] // c['num_heads'])
        normalized = L.LayerNormalization(epsilon=1e-5)(x)
        attended = attn(normalized, normalized, use_causal_mask=causal)
        x = x + attended
        if family == 'vit':
            x = x + L.Dense(c['d_model'])(L.Activation('gelu')(L.Dense(c['mlp_hidden'])(L.LayerNormalization(epsilon=1e-5)(x))))
            outputs = L.Dense(c['output_dim'])(L.Lambda(lambda z: tf.reduce_mean(z, axis=1))(x))
        else:
            residual = L.Flatten()(x)
            y = L.Flatten()(L.LayerNormalization(epsilon=1e-5)(x))
            y = L.Dense(c['mlp_hidden'], activation='gelu')(y)
            y = L.Dense(c['seq_len'] * c['d_model'])(y)
            outputs = L.Dense(c['output_dim'])(y + residual)
    elif family == 'vgg16':
        x = inputs
        for block, count in enumerate((2, 2, 3, 3, 3)):
            channels = c['base_channels'] * (2 ** block)
            for _ in range(count):
                x = L.Conv2D(channels, 3, padding='same', use_bias=False)(x)
                x = L.LayerNormalization(axis=[1, 2, 3], center=False, scale=False)(x)
                x = L.Activation('relu')(x)
            x = L.Conv2D(channels, 3, strides=2, padding='same', use_bias=False)(x)
            x = L.LayerNormalization(axis=[1, 2, 3], center=False, scale=False)(x)
            x = L.Activation('relu')(x)
        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(c['fc_hidden'], activation='gelu')(x)
        x = L.Dense(c['fc_hidden'], activation='gelu')(x)
        outputs = L.Dense(c['num_classes'])(x)
    elif family == 'resnet':
        x = L.Activation('relu')(L.Conv2D(c['base_channels'], 3, strides=2, padding='same', use_bias=False)(inputs))
        for stage, count in enumerate((c['blocks1'], c['blocks2'], c['blocks3'], c['blocks4'])):
            channels = c['base_channels'] * (2 ** stage)
            for index in range(count):
                stride = 2 if stage and index == 0 else 1
                shortcut = x
                y = L.Activation('relu')(L.Conv2D(channels, 3, strides=stride, padding='same', use_bias=False)(x))
                y = L.Conv2D(channels, 3, padding='same', use_bias=False)(y)
                if stride != 1 or int(shortcut.shape[-1]) != channels:
                    shortcut = L.Conv2D(channels, 1, strides=stride, use_bias=False)(shortcut)
                x = L.Activation('relu')(L.Add()([y, shortcut]))
        outputs = L.Dense(c['num_classes'])(L.GlobalAveragePooling2D()(x))
    elif family == 'vae_conv':
        x = L.Activation('silu')(L.Conv2D(c['base_channels'], 3, strides=2, padding='same')(inputs))
        x = L.Activation('silu')(L.Conv2D(c['base_channels'] * 2, 3, strides=2, padding='same')(x))
        mu = L.Conv2D(c['latent_c'], 1)(x)
        logvar = L.Conv2D(c['latent_c'], 1)(x)
        y = L.Activation('silu')(L.Conv2D(c['base_channels'] * 2, 3, padding='same')(mu))
        y = L.UpSampling2D(interpolation='nearest')(y)
        y = L.Activation('silu')(L.Conv2D(c['base_channels'], 3, padding='same')(y))
        y = L.UpSampling2D(interpolation='nearest')(y)
        recon = L.Activation('tanh')(L.Conv2D(c['image_c'], 3, padding='same')(y))
        outputs = L.Flatten()(recon)
    elif family == 'unet':
        x, skips = inputs, []
        for depth in range(c['depth']):
            channels = c['base_channels'] * (2 ** depth)
            x = L.Activation('relu')(L.Conv2D(channels, 3, padding='same')(x))
            x = L.Activation('relu')(L.Conv2D(channels, 3, padding='same')(x))
            skips.append(x)
            x = L.Activation('relu')(L.Conv2D(channels, 3, strides=2, padding='same')(x))
        x = L.Activation('relu')(L.Conv2D(c['base_channels'] * (2 ** (c['depth'] - 1)) * 2, 3, padding='same')(x))
        for depth, skip in reversed(list(enumerate(skips))):
            channels = c['base_channels'] * (2 ** depth)
            x = L.UpSampling2D(interpolation='nearest')(x)
            x = L.Concatenate()([x, skip])
            x = L.Activation('relu')(L.Conv2D(channels, 3, padding='same')(x))
            x = L.Activation('relu')(L.Conv2D(channels, 3, padding='same')(x))
        outputs = L.Flatten()(L.Conv2D(c['image_c'], 1)(x))
    else:
        raise ValueError('Unknown architecture family: ' + family)
    return keras.Model(inputs, outputs)


def _run_tensorflow(config, backend):
    import tensorflow as tf
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    strategy = None
    wanted = {'cpu': 'CPU', 'cuda': 'GPU', 'tpu': 'TPU'}[backend]
    if backend == 'cuda':
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('CUDA requested but TensorFlow GPU is unavailable; refusing CPU fallback')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        device_scope = tf.device('/GPU:0')
    elif backend == 'tpu':
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
        except Exception as exc:
            raise RuntimeError('TensorFlow TPU adapter could not prove TPU placement: ' + str(exc)) from exc
        if strategy.num_replicas_in_sync != 1 or not any('TPU' in d.upper() for d in strategy.extended.worker_devices):
            raise RuntimeError('TensorFlow TPU adapter requires exactly one proven TPU replica')
        device_scope = strategy.scope()
    else:
        device_scope = tf.device('/CPU:0')
    with device_scope:
        model = _tf_model(tf, config)
    if any(wanted not in v.device.upper() for v in model.variables):
        raise RuntimeError('TensorFlow variables are not on requested ' + wanted + ' device')
    inputs = _input_data(config, 'tensorflow', backend)
    if strategy:
        @tf.function
        def forward():
            return strategy.run(model, args=(inputs,), kwargs={'training': False})
    else:
        @tf.function
        def forward():
            with tf.device('/GPU:0' if backend == 'cuda' else '/CPU:0'):
                return model(inputs, training=False)
    def sync():
        output = forward()
        if strategy:
            outputs = strategy.experimental_local_results(output)
            if not outputs or any('TPU' not in value.device.upper() for value in outputs):
                raise RuntimeError('TensorFlow output placement did not prove TPU execution')
            graph = forward.get_concrete_function().graph.as_graph_def()
            nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
            if not any('TPU' in node.op or '_tpu_replicate' in node.attr for node in nodes):
                raise RuntimeError('TensorFlow TPU execution graph proof is missing')
            output = outputs[0]
        if strategy:
            for value in outputs:
                value.numpy()
        else:
            output.numpy()
        return output
    return model, inputs, forward, sync, lambda out: (
        wanted in out.device.upper()), wanted


def _jax_model(config):
    import jax
    import jax.numpy as jnp
    import numpy as np
    family, c = config['family'], config['model_config']
    rng = np.random.default_rng(12031)
    params = {}
    def linear(name, inp, out):
        params[name] = (jnp.asarray((rng.standard_normal((inp, out)) * np.sqrt(2.0 / max(inp, 1))).astype('float32')),
                        jnp.zeros((out,), dtype=jnp.float32))
    def conv(name, inp, out, kernel=3):
        params[name] = (jnp.asarray((rng.standard_normal((kernel, kernel, inp, out)) * np.sqrt(2.0 / (kernel * kernel * inp))).astype('float32')),
                        jnp.zeros((out,), dtype=jnp.float32))
    def dense(name, x):
        w, b = params[name]
        return x @ w + b
    def conv2(name, x, stride=1, padding='SAME'):
        w, b = params[name]
        return jax.lax.conv_general_dilated(x, w, (stride, stride), padding,
                  dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + b
    def norm(x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(var + 1e-5)
    def norm_all(x):
        mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True)
        return (x - mean) * jax.lax.rsqrt(var + 1e-5)
    if family == 'basic_mlp':
        dims = [c['input_dim']] + [c['hidden_dim']] * c['hidden_layers'] + [c['output_dim']]
        for i, (a, b) in enumerate(zip(dims, dims[1:])): linear('l'+str(i), a, b)
    elif family == 'vgg16':
        current = c['image_c']
        for block, count in enumerate((2, 2, 3, 3, 3)):
            channels = c['base_channels'] * (2 ** block)
            for i in range(count):
                conv(f'b{block}_{i}', current, channels); current = channels
            conv(f'd{block}', current, current)
        linear('fc1', current, c['fc_hidden'])
        linear('fc2', c['fc_hidden'], c['fc_hidden'])
        linear('head', c['fc_hidden'], c['num_classes'])
    elif family == 'resnet':
        conv('stem', c['image_c'], c['base_channels'])
        current = c['base_channels']
        for stage, count in enumerate((c['blocks1'], c['blocks2'], c['blocks3'], c['blocks4'])):
            channels = c['base_channels'] * (2 ** stage)
            for block in range(count):
                stride = 2 if stage and block == 0 else 1
                conv(f's{stage}_{block}_a', current, channels)
                conv(f's{stage}_{block}_b', channels, channels)
                if stride != 1 or current != channels: conv(f's{stage}_{block}_skip', current, channels, 1)
                current = channels
        linear('head', current, c['num_classes'])
    elif family == 'vae_conv':
        conv('e1', c['image_c'], c['base_channels']); conv('e2', c['base_channels'], c['base_channels']*2)
        conv('mu', c['base_channels']*2, c['latent_c'], 1); conv('lv', c['base_channels']*2, c['latent_c'], 1)
        conv('d1', c['latent_c'], c['base_channels']*2); conv('d2', c['base_channels']*2, c['base_channels'])
        conv('d3', c['base_channels'], c['image_c'])
    elif family in ('transformer', 'causal_transformer', 'vit'):
        d, heads = c['d_model'], c['num_heads']
        if family != 'vit':
            params['embed'] = jnp.asarray((rng.standard_normal((c['vocab_size'], d)) * .05).astype('float32'))
            for name in ('q', 'k', 'v', 'o'): linear(name, d, d)
        else:
            for name in ('q', 'k', 'v', 'o'): linear(name, d, d)
        if family == 'vit':
            linear('f1', d, c['mlp_hidden']); linear('f2', c['mlp_hidden'], d); linear('head', d, c['output_dim'])
        else:
            size = c['seq_len'] * d
            linear('f1', size, c['mlp_hidden']); linear('f2', c['mlp_hidden'], size); linear('head', size, c['output_dim'])
    elif family == 'unet':
        current = c['image_c']
        for depth in range(c['depth']):
            channels = c['base_channels'] * (2 ** depth)
            conv(f'd{depth}a', current, channels); conv(f'd{depth}b', channels, channels); conv(f'd{depth}down', channels, channels)
            current = channels
        conv('bottle', current, current * 2); current *= 2
        for depth in reversed(range(c['depth'])):
            channels = c['base_channels'] * (2 ** depth)
            conv(f'u{depth}a', current + channels, channels); conv(f'u{depth}b', channels, channels); current = channels
        conv('out', current, c['image_c'], 1)
    def attention(x, causal=False):
        n, length, d = x.shape
        normalized = norm(x)
        q = dense('q', normalized).reshape(n, length, heads, d//heads).transpose(0, 2, 1, 3)
        k = dense('k', normalized).reshape(n, length, heads, d//heads).transpose(0, 2, 1, 3)
        v = dense('v', normalized).reshape(n, length, heads, d//heads).transpose(0, 2, 1, 3)
        logits = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d//heads)
        if causal:
            logits = jnp.where(jnp.triu(jnp.ones((length, length), dtype=bool), 1)[None, None], -1e9, logits)
        weights = jax.nn.softmax(logits, axis=-1)
        return dense('o', (weights @ v).transpose(0, 2, 1, 3).reshape(n, length, d))
    def forward(x):
        if family == 'basic_mlp':
            for i in range(c['hidden_layers']): x = jax.nn.gelu(dense('l'+str(i), x))
            return dense('l'+str(c['hidden_layers']), x)
        if family == 'vgg16':
            for block, count in enumerate((2, 2, 3, 3, 3)):
                for i in range(count): x = jax.nn.relu(norm_all(conv2(f'b{block}_{i}', x)))
                x = jax.nn.relu(norm_all(conv2(f'd{block}', x, 2)))
            x = jnp.mean(x, axis=(1, 2))
            x = jax.nn.gelu(dense('fc1', x))
            return dense('head', jax.nn.gelu(dense('fc2', x)))
        if family == 'resnet':
            x = jax.nn.relu(conv2('stem', x, 2))
            current = c['base_channels']
            for stage, count in enumerate((c['blocks1'], c['blocks2'], c['blocks3'], c['blocks4'])):
                channels = c['base_channels'] * (2 ** stage)
                for block in range(count):
                    stride = 2 if stage and block == 0 else 1
                    residual = x
                    y = jax.nn.relu(conv2(f's{stage}_{block}_a', x, stride))
                    y = conv2(f's{stage}_{block}_b', y)
                    if stride != 1 or current != channels: residual = conv2(f's{stage}_{block}_skip', residual, stride, 'VALID' if False else 'SAME')
                    x = jax.nn.relu(y + residual); current = channels
            return dense('head', jnp.mean(x, axis=(1, 2)))
        if family == 'vae_conv':
            x = jax.nn.silu(conv2('e1', x, 2)); x = jax.nn.silu(conv2('e2', x, 2))
            mu, lv = conv2('mu', x), conv2('lv', x)
            y = jax.nn.silu(conv2('d1', mu))
            y = jax.image.resize(y, (1, y.shape[1]*2, y.shape[2]*2, y.shape[3]), method='nearest')
            y = jax.nn.silu(conv2('d2', y))
            y = jax.image.resize(y, (1, y.shape[1]*2, y.shape[2]*2, y.shape[3]), method='nearest')
            recon = jnp.tanh(conv2('d3', y))
            return recon.reshape(-1)
        if family in ('transformer', 'causal_transformer'):
            x = params['embed'][x]
            x = x + attention(x, c['causal'])
            residual = x.reshape(1, -1)
            y = norm(x).reshape(1, -1)
            y = dense('f2', jax.nn.gelu(dense('f1', y))) + residual
            return dense('head', y)
        if family == 'vit':
            x = x.reshape(1, c['num_tokens'], c['d_model'])
            x = x + attention(x, c['causal'])
            y = x + dense('f2', jax.nn.gelu(dense('f1', norm(x))))
            return dense('head', jnp.mean(y, axis=1))
        skips = []
        for depth in range(c['depth']):
            x = jax.nn.relu(conv2(f'd{depth}a', x))
            x = jax.nn.relu(conv2(f'd{depth}b', x))
            skips.append(x)
            x = jax.nn.relu(conv2(f'd{depth}down', x, 2))
        x = jax.nn.relu(conv2('bottle', x))
        for depth, skip in zip(reversed(range(c['depth'])), reversed(skips)):
            x = jax.image.resize(x, (1, skip.shape[1], skip.shape[2], x.shape[-1]), method='nearest')
            x = jnp.concatenate((x, skip), axis=-1)
            x = jax.nn.relu(conv2(f'u{depth}a', x))
            x = jax.nn.relu(conv2(f'u{depth}b', x))
        return conv2('out', x).reshape(-1)
    return params, forward


def _run_jax(config, backend):
    import jax
    import numpy as np
    wanted = {'cpu': 'cpu', 'cuda': 'gpu', 'tpu': 'tpu'}[backend]
    devices = jax.devices(wanted)
    if not devices or devices[0].platform != wanted:
        raise RuntimeError('JAX requested device unavailable; refusing fallback')
    device = devices[0]
    params, model = _jax_model(config)
    params = jax.device_put(params, device)
    inputs = _input_data(config, 'jax', backend)
    inputs = jax.device_put(inputs, device)
    forward = jax.jit(model)
    def check(out):
        return all(d.platform == wanted for d in out.devices())
    def sync(output):
        jax.block_until_ready(output)
        return output
    return params, inputs, lambda: forward(inputs), sync, check, str(device)


def run(framework, backend, config):
    _validate_config(config)
    if framework not in ('mimir', 'pytorch', 'tensorflow', 'jax'):
        raise ValueError('Unknown framework: ' + framework)
    if backend not in ('cpu', 'cuda', 'tpu'):
        raise ValueError('Unsupported backend: ' + backend)
    if framework == 'mimir':
        raise ValueError('Mímir execution is provided by the suite native executable')
    _seed(framework, backend, int(config.get('seed', 12031)))
    setup_started = time.perf_counter()
    if framework == 'pytorch':
        _model, _inputs, forward, sync, placed, device_name = _run_pytorch(config, backend)
        parameter_count = sum(parameter.numel() for parameter in _model.parameters())
        def array(value):
            return value.detach().cpu().numpy()
    elif framework == 'tensorflow':
        model, _inputs, forward, sync, placed, device_name = _run_tensorflow(config, backend)
        parameter_count = model.count_params()
        def array(value):
            return value.numpy()
    else:
        import jax
        _params, _inputs, forward, sync, placed, device_name = _run_jax(config, backend)
        parameter_count = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(_params))
        def array(value):
            return value
    setup_s = time.perf_counter() - setup_started

    started = time.perf_counter()
    output = sync(forward())
    cold_forward_ms = (time.perf_counter() - started) * 1000
    if not placed(output):
        raise RuntimeError(f'{framework} output placement cannot prove requested {backend} device')
    import numpy as np
    actual = np.asarray(array(output))
    if actual.size != config['expected_output_size']:
        raise RuntimeError(f'Wrong output size: expected {config["expected_output_size"]}, got {actual.size}')
    if not np.all(np.isfinite(actual)):
        raise RuntimeError('Non-finite architecture output')
    for _ in range(config['warmup']):
        output = sync(forward())
        if not placed(output):
            raise RuntimeError('Output left the requested device during warmup')
    samples = []
    for _ in range(config['iterations']):
        started = time.perf_counter()
        output = sync(forward())
        samples.append((time.perf_counter() - started) * 1000)
        if not placed(output):
            raise RuntimeError('Output left the requested device during timed inference')
        actual = np.asarray(array(output))
        if actual.size != config['expected_output_size'] or not np.all(np.isfinite(actual)):
            raise RuntimeError('Timed forward produced invalid output')
    samples.sort()
    median = statistics.median(samples)
    p95 = samples[max(0, math.ceil(0.95 * len(samples)) - 1)]
    return dict(framework=framework, backend=backend, case=config['case'], family=config['family'],
                architecture=config['architecture'], status='measured',
                benchmark_kind='architecture', claim='fixed-input forward inference only',
                mode='inference forward', version=str(_version(framework)), device=device_name,
                setup_s=setup_s, cold_forward_ms=cold_forward_ms,
                parameter_count=parameter_count, forward_samples_ms=samples,
                forward_median_ms=median, forward_p95_ms=p95,
                input_size=config['input_size'], token_count=config['token_count'],
                output_size=config['expected_output_size'], first_output=actual.reshape(-1).tolist())


def _version(framework):
    if framework == 'pytorch':
        import torch
        return torch.__version__
    if framework == 'tensorflow':
        import tensorflow as tf
        return tf.__version__
    import jax
    return jax.__version__


def main(framework, backend, config_path, result_path):
    config = json.loads(Path(config_path).read_text())
    try:
        result = run(framework, backend, config)
    except Exception as exc:
        result = dict(framework=framework, backend=backend, case=config.get('case'),
                      benchmark_kind='architecture', status='error', error=str(exc),
                      traceback=traceback.format_exc())
    Path(result_path).write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get('status') == 'measured' else 1


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('usage: colab_architecture_worker.py FRAMEWORK BACKEND CONFIG RESULT')
    raise SystemExit(main(*sys.argv[1:]))
