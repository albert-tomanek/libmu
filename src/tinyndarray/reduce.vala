internal int ResolveAxis(int axis, size_t ndim, string name) {
    // Resolve negative
    int ndim_i = (int) ndim;
    if (axis < 0) {
        axis = ndim_i + axis;
    }
    // Check range
    if (axis < 0 || ndim_i <= axis) {
        error(@"Invalid axes for $name ($ndim vs $axis)");
    }
    return axis;
}

int[] _sorted(int[] arr)
{
    // Sort arr from largest to smallest
    int[] ret = new int[arr.length];

    int last_max = int.MAX;
    for (int out_i = 0; out_i < arr.length; out_i++) {
        int max_i = 0;
        for (int in_i = 0; in_i < arr.length; in_i++) {
            if (arr[in_i] > arr[max_i] && arr[in_i] < last_max) {
                max_i = in_i;
            }
        }

        ret[out_i] = last_max = arr[max_i];     // So last_max will be steadily decreasing
    }

    return ret;
}

int[] _reversed(int[] arr)
{
    // Reverse arr
    int[] ret = new int[arr.length];
    for (int i = 0; i < arr.length; i++) {
        ret[i] = arr[arr.length - i - 1];
    }
    return ret;
}

internal int[] ResolveAxisSort(int[] axes, size_t ndim, string name,
                        bool sort, bool sort_order_normal = true) {
    // Resolve for each
    int[] ret_axes = {};

    foreach (int axis in axes) {
        ret_axes += (ResolveAxis(axis, ndim, name));
    }
    // Sort axes
    if (sort) {
        if (sort_order_normal) {
            // Normal order
            _reversed(_sorted(ret_axes));
        } else {
            // Inverse order
            _sorted(ret_axes);
        }
    }
    return ret_axes;
}

internal int[] CheckReductable(int[] shape, int[] axes,
                             bool keepdims) {
    // Mark reduction axes
    var mark = new bool[shape.length];
    int n_shape = (int) shape.length;
    foreach (int axis in axes) {
        if (0 <= axis && axis < n_shape) {
            mark[axis] = true;
        } else {
            error("Axis %d out of bounds for shape %s.", axis, Mu.print_shape(shape));
        }
    }

    if (keepdims) {
        // Pick up unmarked dimension
        int[] ret_shape_pad = {};
        for (size_t i = 0; i < mark.length; i++) {
            if (mark[i]) {
                ret_shape_pad += (1);
            } else {
                ret_shape_pad += (shape[i]);
            }
        }
        return ret_shape_pad;
    } else {
        // No necessary
        return {};
    }
}

internal int[] ComputeReduceSizes(int[] src_shape, size_t axis, out int n_upper, out int n_lower, out int n_reduce) {
    // Compute result shape
    int[] ret_shape = {};
    for (size_t dim = 0; dim < src_shape.length; dim++) {
        if (dim != axis) {
            ret_shape += (src_shape[dim]);
        }
    }
    if (ret_shape.length == 0) {  // For all reduction
        ret_shape += (1);
    }

    // Compute sizes
    n_upper = 1;
    n_lower = 1;
    n_reduce = 0;
    for (size_t dim = 0; dim < src_shape.length; dim++) {
        // Sizes
        if (dim < axis) {
            n_upper *= src_shape[dim];
        } else if (axis < dim) {
            n_lower *= src_shape[dim];
        } else {
            n_reduce = src_shape[dim];
        }
    }

    // Return
    return ret_shape;
}

float ReduceAxisAll([CCode(array_length = false)] float[] data, size_t size, float init_v, F reduce_op) {
    RunParallelWithReduceFn op = (i) => data[i];
    float ret = RunParallelWithReduce((int) size, op,
                                reduce_op, init_v);
    return ret;
}

delegate void ReduceAxisOne_ReduceFn(int u_idx);

Mu.Array ReduceAxisOne(Mu.Array src, size_t axis, float init_v,
                      F reduce_op) {
    int[] src_shape = src.shape;

    // Compute sizes
    int n_upper, n_lower, n_reduce;
    int[] ret_shape = ComputeReduceSizes(src_shape, axis, out n_upper, out n_lower, out n_reduce);

    // Create result array with fill
    Mu.Array ret = Mu.mul(Mu.ones(ret_shape), Mu.scalar(init_v));

    float[] src_data = ((float[]) src.bytes.data)[src.start:];
    float[] ret_data = ((float[]) ret.bytes.data);

    // Reduce function
    ReduceAxisOne_ReduceFn reduce = (u_idx) => {
        int ret_idx_base = u_idx * n_lower;
        int src_idx_base0 = u_idx * n_reduce * n_lower;
        for (int redu_idx = 0; redu_idx < n_reduce; redu_idx++) {
            int src_idx_base1 = src_idx_base0 + redu_idx * n_lower;
            for (int l_idx = 0; l_idx < n_lower; l_idx++) {
                // Reduce
                float r = ret_data[ret_idx_base + l_idx];
                float s = src_data[src_idx_base1 + l_idx];
                r = reduce_op(r, s);
                ret_data[ret_idx_base + l_idx] = r;
            }
        }
    };

    // TODO: Run parallel for `axis == 0` (means `n_upper == 1`)

#if 0  // Run in parallel
    RunParallel(n_upper, reduce);
#else  // Run sequentially
    for (int u_idx = 0; u_idx < n_upper; u_idx++) {
        reduce(u_idx);
    }
#endif
    return ret;
}


Mu.Array ReduceAxis(Mu.Array src, int[] axes_raw, bool keepdims,
                   float init_v, F reduce_op) {
    if (axes_raw.length == 0) {
        // No int[] -> Reduce all
        float ret_v = ReduceAxisAll(((float[]) src.bytes.data)[src.start:], Mu.shape_length(src.shape), init_v, reduce_op);
        Mu.Array ret = Mu.scalar(ret_v);
        // Reshape for keepdims
        if (keepdims) {
            int[] ret_shape = new int[src.shape.length];
            for (int i = 0; i < ret_shape.length; i++) { ret_shape[i] = 1; }
            ret = ret.reshape(ret_shape);
        }
        return ret;
    } else {
        // Resolve axes (sort: on)
        int[] axes = ResolveAxisSort(axes_raw, src.ndim, "Reduce", true);

        // Check it is possible to reduce.
        int[] src_shape = src.shape;
        int[] ret_shape_pad = CheckReductable(src_shape, axes, keepdims);

        // Reduce axes one by one
        Mu.Array ret = src;
        for (size_t i = 0; i < axes.length; i++) {
            // From back
            size_t axis = (size_t) axes[axes.length - i - 1];
            // Reduce
            ret = ReduceAxisOne(ret, axis, init_v, reduce_op);
        }

        // Reshape for keepdims
        if (keepdims) {
            ret = ret.reshape(ret_shape_pad);
        }
        return ret;
    }
}
