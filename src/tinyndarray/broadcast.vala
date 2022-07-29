// ------------------ Utilities for NdArray (Broadcast common) -----------------

internal int[] CheckBroadcastable(int[] l_shape, int[] r_shape) {
    // We assuming left array has deeper shape than right one.
    if (l_shape.length < r_shape.length) {
        return CheckBroadcastable(r_shape, l_shape);  // Swap
    }
    // `l_shape.length` is maximum depth.

    // Check empty
    if (r_shape.length == 0 || (r_shape.length == 1 && r_shape[0] == 0)) {
        error("Can't broadcast an empty array (of shape %s).", Mu.print_shape(r_shape));
    }

    // Compute broadcasted shape
    int[] shape = new int[l_shape.length];
    size_t r_offset = l_shape.length - r_shape.length;
    for (size_t i = 0; i < l_shape.length; i++) {
        if (i < r_offset) {
            shape[i] = l_shape[i];
        } else {
            int l = l_shape[i];
            int r = r_shape[i - r_offset];
            if (l == r) {
                shape[i] = l;  // no broadcast
            } else if (l == 1) {
                shape[i] = r;  // left broadcast
            } else if (r == 1) {
                shape[i] = l;  // right broadcast
            } else {
                error("Shapes %s and %s are not compatible in dimention %d.", Mu.print_shape(l_shape), Mu.print_shape(r_shape), (int) i);
            }
        }
    }
    return shape;
}

internal int[] PadShape(int[] shape, size_t size) {
    if (size < shape.length) {
        error("Shape %s is too long to pad to %d dimensions.", Mu.print_shape(shape), (int) size);
    }
    size_t n_pad = size - shape.length;
    int[] ret_shape = new int[size];
    for (size_t i = 0; i < ret_shape.length; i++) {
        if (i < n_pad) {
            ret_shape[i] = 1;
        } else {
            ret_shape[i] = shape[i - n_pad];
        }
    }
    return ret_shape;
}

internal size_t ReduceShapesBroadcast(ref int[] ret_shape, ref int[] l_shape,
                                    ref int[] r_shape, size_t depth_offset)
requires(ret_shape.length == l_shape.length)
requires(ret_shape.length == r_shape.length)
{

    // Remove meaningless dimensions.
    int[] ret_shape_cleaned = {}, l_shape_cleaned = {}, r_shape_cleaned = {};    
    int size_pool = 1;
    size_t depth = 0;
    for (; depth < ret_shape.length - depth_offset; depth++) {
        if (l_shape[depth] == r_shape[depth]) {
            // Store
            size_pool *= l_shape[depth];
        } else {
            // Pop
            if (size_pool != 1) {
                ret_shape_cleaned += (size_pool);
                l_shape_cleaned += (size_pool);
                r_shape_cleaned += (size_pool);
                size_pool = 1;
            }
            // Through current dimension
            ret_shape_cleaned += (ret_shape[depth]);
            l_shape_cleaned += (l_shape[depth]);
            r_shape_cleaned += (r_shape[depth]);
        }
    }
    // Pop
    if (size_pool != 1 || ret_shape_cleaned.length == 0) {
        ret_shape_cleaned += (size_pool);
        l_shape_cleaned += (size_pool);
        r_shape_cleaned += (size_pool);
    }
    // Store actual depth count
    size_t n_depth = ret_shape_cleaned.length;
    // Pass through included in `depth_offset`.
    for (; depth < ret_shape.length; depth++) {
        ret_shape_cleaned += (ret_shape[depth]);
        l_shape_cleaned += (l_shape[depth]);
        r_shape_cleaned += (r_shape[depth]);
    }
    // Return
    ret_shape = ret_shape_cleaned;
    l_shape = l_shape_cleaned;
    r_shape = r_shape_cleaned;
    return n_depth;
}

internal int[] ComputeChildSizes(int[] shape) {
    size_t n_shape = shape.length;
    if (n_shape == 0) {
        return {};
    }
    // Compute child sizes from back (the number of children for each dimension)
    int[] child_sizes = new int[n_shape];
    int size = 1;
    for (size_t depth = n_shape - 1; 0 < depth; depth--) {
        child_sizes[depth] = size;
        size *= shape[depth];
    }
    child_sizes[0] = size;
    return child_sizes;
}


delegate float F(float p, float q);

void ApplyOpBroadcastImpl([CCode(array_length = false)] float[] ret_data,
                          [CCode(array_length = false)] float[] l_data,
                          [CCode(array_length = false)] float[] r_data,
                          int[] ret_shape, int ret_size,
                          int[] l_steps,
                          int[] r_steps,
                          size_t start_depth, size_t n_depth,
                          int ret_step, F op) {
    // Create stacks and counter
    int[] ret_cnts = new int[n_depth];
    int[] l_idx_stack = new int[n_depth], r_idx_stack = new int[n_depth];
    size_t depth = start_depth; 
    int l_idx = 0;
    int r_idx = 0;

    for (int ret_idx = 0; ret_idx < ret_size; ret_idx += ret_step) {
        // Go down
        for (; depth < n_depth; depth++) {
            l_idx_stack[depth] = l_idx;  // Push stack
            r_idx_stack[depth] = r_idx;
        }

        // Operate
        ret_data[ret_idx] = op(l_data[l_idx], r_data[r_idx]);

        // Go up and count
        for (; start_depth < depth; depth--) {
            size_t prev_d = depth - 1;
            ret_cnts[prev_d]++;        // Count up
            l_idx += l_steps[prev_d];  // Forward index
            r_idx += r_steps[prev_d];
            if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                break;  // Continue normally
            }
            // Go upper depth
            ret_cnts[prev_d] = 0;         // Clear count
            l_idx = l_idx_stack[prev_d];  // Pop stack
            r_idx = r_idx_stack[prev_d];
        }
    }
}

void ApplyOpBroadcast(Mu.Array ret, Mu.Array lhs, Mu.Array rhs,
                      size_t depth_offset, int ret_step, F op) {
    int[] ret_shape = ret.shape;

    // Pre-compute padded shapes
    int[] l_shape = PadShape(lhs.shape, ret_shape.length);
    int[] r_shape = PadShape(rhs.shape, ret_shape.length);

    // Pre-compute reduced shapes
    size_t n_depth =
            ReduceShapesBroadcast(ref ret_shape, ref l_shape, ref r_shape, depth_offset);

    // Pre-compute child sizes
    int[] ret_child_sizes = ComputeChildSizes(ret_shape);
    int[] l_child_sizes = ComputeChildSizes(l_shape);
    int[] r_child_sizes = ComputeChildSizes(r_shape);

    // Pre-compute steps
    int[] l_steps = {}, r_steps = {};
    for (size_t depth = 0; depth < n_depth; depth++) {
        int l_s = l_shape[depth];
        int r_s = r_shape[depth];
        int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        l_steps += (l_step);
        r_steps += (r_step);
    }

#if 0  // Run in parallel
    RunParallel(ret_shape[0], [&](int i) {
        int ret_size = static_cast<int>(ret.length) / ret_shape[0];
        ApplyOpBroadcastImpl(ret.data() + ret_child_sizes[0] * i,
                             lhs.data() + l_steps[0] * i,
                             rhs.data() + r_steps[0] * i, ret_shape, ret_size,
                             l_steps, r_steps, 1, n_depth, ret_step, op);
    });
#else  // Run sequentially
    ApplyOpBroadcastImpl(((float[]) ret.bytes.data), ((float[]) lhs.bytes.data)[lhs.start:], ((float[]) rhs.bytes.data)[rhs.start:], ret_shape,
                         Mu.shape_length(ret.shape), l_steps, r_steps, 0,
                         n_depth, ret_step, op);
#endif
}

Mu.Array ApplyDualOp(Mu.Array lhs, Mu.Array rhs, F op) {
    /*if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        NdArray ret(lhs.shape());
        // Simply apply all
        ApplyOpSimple(ret, lhs, rhs, op);
        return ret;
    } else*/ {
        // Check it is possible to broadcast
        int[] ret_shape = CheckBroadcastable(lhs.shape, rhs.shape);
        // Apply broadcast
        var ret = Mu.zeros(ret_shape, lhs.dtype);
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, op);
        return ret;
    }
}

