from keras.layers import Layer
from keras import backend as K
from keras.layers import Layer, InputSpec

class HierarchicalMerge(Layer):
    """
        Inputs: [N-D, [2-D, N-D], ... ]
        axis-0: batch
        axis-1:
            inputs 2k: timesteps
            inputs 2k+1: position/boundary of inputs 2k mapped to inputs 2k+2
        axis-2,3,...,N-1: features    
    """
    def __init__(self, **kwargs):
        super(HierarchicalMerge, self).__init__(**kwargs)
        self.input_spec = None

    def build(self, input_shape):
        super(HierarchicalMerge, self).build(input_shape)
        self.input_spec = None

    def merge_fn(self, inputs):
        raise NotImplementedError

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list): inputs = [inputs]

        shapes = [K.shape(x) for x in inputs]
        batch_size = shapes[0][0]
        x_steps = [s[1] for s in shapes[::2]]
        ndims = K.ndim(inputs[0])
        cur_poses = [K.arange(batch_size * x_steps[0])]
        batch_multiplier = K.arange(batch_size)
        for x_pos in [K.flatten(x_pos[:, :-1] + (batch_multiplier * x_step)[:, None]) 
                        for x_step, x_pos in zip(x_steps, inputs[1::2])]:
            cur_poses += [K.sum(K.cast(x_pos[None,:] <= cur_poses[-1][:,None], dtype='int32'),axis=-1) - 1]
        y = [inputs[0]] + [
                K.reshape(
                    K.gather(
                        K.reshape(x, shape=(-1,)+tuple(shape[j] for j in range(2,ndims))), 
                        indices),
                    shape=(batch_size,x_steps[0])+tuple(shape[j] for j in range(2,ndims))
                )
            for indices, x, shape in zip(cur_poses[1:], inputs[2::2], shapes[2::2])]
        y = self.merge_fn(y)
        return y

class HierarchicalConcatenate(HierarchicalMerge):
    def __init__(self, axis=-1, **kwargs):
        super(HierarchicalConcatenate, self).__init__(**kwargs)
        self.axis = axis 

    def merge_fn(self, inputs):
        return K.concatenate(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        return input_shape[0][:-1] + (
            sum((s[-1] for s in input_shape[::2])) if all((s[-1] for s in input_shape[::2])) else None,)

    def get_config(self):
        return dict(**{
            'axis': self.axis
        }, **super(HierarchicalConcatenate, self).get_config())


class ReduceOp(Layer):
    def __init__(self, axis, keepdims=False, **kwargs):
        super(ReduceOp, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        if self.axis is None:
            axes = list(range(len(input_shape)))
        else:
            axes = list(self.axis) if isinstance(self.axis, (tuple, list)) else [self.axis]
            axes = [axis if axis > 0 else len(input_shape) + axis for axis in axes]

        output_shape = list(input_shape)
        for axis in axes:
            output_shape[axis] = 1
        if not self.keepdims:
            output_shape = [s for i, s in enumerate(output_shape) if i not in axes]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'axis': self.axis,
            'keepdims': self.keepdims
        }
        return dict(**super(ReduceOp, self).get_config(), **config)


class Max(ReduceOp):
    def call(self, x, mask=None):
        if mask is None:
            return K.max(x, axis=self.axis, keepdims=self.keepdims)
        mask = K.cast(mask, dtype=K.dtype(x))
        mask = K.expand_dims(mask, -1)
        x_min = K.min(x, axis=self.axis, keepdims=self.keepdims)
        x = x * mask + (x_min - K.epsilon()) * (1 - mask)
        return K.max(x, axis=self.axis, keepdims=self.keepdims)


class Min(ReduceOp):
    def call(self, x, mask=None):
        if mask is None:
            return K.min(x, axis=self.axis, keepdims=self.keepdims)
        mask = K.cast(mask, dtype=K.dtype(x))
        mask = K.expand_dims(mask, -1)
        x_max = K.max(x, axis=self.axis, keepdims=self.keepdims)
        x = x * mask + (x_max + K.epsilon()) * (1 - mask)
        return K.min(x, axis=self.axis, keepdims=self.keepdims)


class Avg(ReduceOp):
    def __init__(self, **kwargs):
        super(Avg, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        if mask is None:
            return K.mean(x, axis=self.axis, keepdims=self.keepdims)

        mask = K.cast(mask, dtype=K.dtype(x))
        mask = K.expand_dims(mask, -1)
        return K.sum(x * mask,
                     axis=self.axis,
                     keepdims=self.keepdims) / (
                       K.sum(mask,
                             axis=self.axis,
                             keepdims=self.keepdims) + K.epsilon())

    def compute_mask(self, x, mask=None):
        return None

class StandardDeviation(ReduceOp):
    def __init__(self, **kwargs):
        super(StandardDeviation, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        if mask is None:
            return K.std(x, axis=self.axis, keepdims=self.keepdims)

        mask = K.cast(mask, dtype=K.dtype(x))
        mask = K.expand_dims(mask, -1)

        masked_x = x * mask
        denominator = K.sum(mask, axis=self.axis, keepdims=self.keepdims) + K.epsilon()

        mean_x = K.sum(x * mask, axis=self.axis, keepdims=self.keepdims) / denominator
        mean_x2 = K.sum(K.square(masked_x), axis=self.axis, keepdims=self.keepdims) / denominator

        return K.sqrt(mean_x2 - K.square(mean_x))

    def compute_mask(self, x, mask=None):
        return None

class ReduceSum(ReduceOp):
    def call(self, inputs, **kwargs):
        return K.sum(inputs, axis=self.axis, keepdims=self.keepdims)

class Broadcasting(Layer):
    """
    Broadcasting axis(axes) of inputs[0] into shape of inputs[1]
    """

    def __init__(self, axis=None, **kwargs):
        super(Broadcasting, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        if self.axis is not None:
            axes = self.axis if isinstance(self.axis, list) else [self.axis]
            for i, axis in enumerate(axes):
                if axis < 0: axes[i] = K.ndim(inputs[0]) + axis
            shape_0 = K.shape(inputs[0])
            shape_1 = K.shape(inputs[1])
            out_shape = tuple(((shape_0 if axis not in axes else shape_1)[axis] for axis in range(K.ndim(inputs[0]))))
        else:
            out_shape = K.shape(inputs[1])
        return inputs[0] + K.zeros(out_shape, dtype=K.dtype(inputs[0]))

    def compute_output_shape(self, input_shape):
        if self.axis is not None:
            shape_0, shape_1 = input_shape
            axes = self.axis if isinstance(self.axis, list) else [self.axis]
            for i, axis in enumerate(axes):
                if axis < 0: axes[i] = len(input_shape[0]) + axis
            out_shape = tuple(((shape_0 if axis not in axes else shape_1)[axis] for axis in range(len(shape_0))))
            return out_shape
        return input_shape[1]

    def get_config(self):
        return dict(**{'axis': self.axis}, **super(Broadcasting, self).get_config())

class LocalPooling1D(Layer):
    def __init__(self, **kwargs):
        super(LocalPooling1D, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]

    def call(self, inputs, mask=None):
        x, x_pos = inputs
        x_shape = K.shape(x)
        batch_size, n_steps, n_channels = x_shape[0], x_shape[1], x_shape[2]
        x_pos = x_pos + (K.arange(batch_size, dtype='int32') * n_steps)[:, None]
        x_bound = K.concatenate([x_pos[:, :-1, None], x_pos[:, 1:, None]], axis=2)
        x_bound = K.reshape(x_bound, shape=(-1, 2))
        x = K.reshape(x, shape=(-1, n_channels))
        mask = K.reshape(mask, shape=(-1, 1)) if mask is not None else None
        y = K.map_fn(
            lambda p, x=x, mask=mask: self.pool_func(x[p[0]:p[1]], mask[p[0]:p[1]] if mask is not None else None),
            x_bound, dtype='float32')
        y = K.reshape(y, shape=(batch_size, -1, n_channels))
        return y

    def pool_func(self, x, mask=None, **kwargs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        x_shape, x_pos_shape = input_shape
        return x_shape[0], x_pos_shape[1] - 1 if x_pos_shape[1] else None, x_shape[2]

    def compute_mask(self, x, mask=None):
        return None

class LocalMaxPooling1D(LocalPooling1D):
    def pool_func(self, x, mask=None, **kwargs):
        if mask is not None:
            mask = K.cast(mask, dtype=K.dtype(x))
            x_min = K.min(x, axis=0, keepdims=True)
            x = x * mask + (x_min - K.epsilon()) * (1 - mask)
        return K.max(x, axis=0)

        return dict(**{'axis': self.axis}, **super().get_config())


class SumToOne(Layer):
    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        return dict(**{'axis': self.axis}, **super().get_config())

    def call(self, inputs, **kwargs):
        return inputs / (K.sum(inputs, axis=self.axis, keepdims=True) + K.epsilon())