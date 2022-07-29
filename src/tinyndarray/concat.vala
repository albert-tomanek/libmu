internal void CheckConcatenatable(Mu.Array[] xs, int axis) {
    // Check empty
    if (xs.length == 0) {
        error("No arrays were passed to Mu.concatenate.");

    }
    // Check same shape except axis dimension
    int[] fst_shape = xs[0].shape;
    Mu.DType fst_dtype = xs[0].dtype;
    size_t axis_l = (size_t)(axis);

    for (size_t i = 1; i < xs.length; i++) {
        int[] cur_shape = xs[i].shape;
        if (xs[i].dtype != fst_dtype) {
            error(@"All arrays must have the same dtype for concatenation. $(xs[i].dtype) != $fst_dtype");
        }
        // Check the size of shapes
        if (fst_shape.length != cur_shape.length) {
            error("Array shapes are incompatible for concatenation. %s != %s",
                Mu.print_shape(cur_shape),
                Mu.print_shape(fst_shape));
        }
        // Check dimensions except axis
        for (size_t j = 0; j < fst_shape.length; j++) {
            if (j == axis_l) {
                continue;
            }
            if (fst_shape[j] != cur_shape[j]) {
                error("Array shapes %s and %s are incompatible for concatenation in axis %d.",
                    Mu.print_shape(cur_shape),
                    Mu.print_shape(fst_shape),
                    (int) j);
            }
        }
    }
}

internal void ComputeConcatSizes(
    Mu.Array[] xs, int axis,
    out int[] ret_shape, out int n_upper, out int[] n_lowers, out int xs_size, out int[] concat_offsets, out int concat_step
) {
    xs_size = xs.length;

    int[] fst_shape = xs[0].shape;
    var src_s_iter0 = 0;
    var src_s_iter1 = 0 + axis;
    var src_s_iter2 = 0 + axis + 1;
    var src_s_iter3 = fst_shape.length;

    // Upper common size
    n_upper = 1;
    for (int i = src_s_iter0; i < src_s_iter1; i++) {
        n_upper *= fst_shape[i];
    }
    // Lower size depends on each sources
    int[] _n_lowers = {};
    foreach (Mu.Array x in xs) {
        _n_lowers += ((int)(Mu.shape_length(x.shape)) / n_upper);
    }
    // Result indices of concatenation
    int[] _concat_offsets = {};
    int n_lower_accum = 0;
    foreach (int n_lower in _n_lowers) {
        _concat_offsets += (n_lower_accum);
        n_lower_accum += n_lower;
    }
    // Step of concatenating dimension
    concat_step = n_lower_accum;

    // Concatenating dimensions
    int concat_dim = 0;
    size_t axis_l = (size_t)(axis);
    foreach (Mu.Array x in xs) {
        concat_dim += x.shape[axis_l];
    }

    // Create result shape
    ret_shape = fst_shape[src_s_iter0:src_s_iter1].copy();  // Upper
    ret_shape = _append(ret_shape, {concat_dim});  // Concatenating dimension
    ret_shape = _append(ret_shape, fst_shape[src_s_iter2:src_s_iter3]);  // Lower

    n_lowers = _n_lowers;
    concat_offsets = _concat_offsets;
}

internal delegate void ConcatenateNdArray_CopyLowerFunc(size_t concat_idx, int ret_idx_base0, int src_idx_base);

internal Mu.Array ConcatenateNdArray(Mu.Array[] xs, int axis) {
    // Resolve axes
    axis = ResolveAxis(axis, xs[0].ndim, "Concatenate");
    // Check it is possible to concatenate
    CheckConcatenatable(xs, axis);

    // Compute concat sizes
    int[] ret_shape;
    int n_upper;
    int[] n_lowers;
    size_t n_concat;
    int[] concat_offsets;
    int concat_step;
    ComputeConcatSizes(xs, axis, out ret_shape, out n_upper, out n_lowers, out n_concat, out concat_offsets, out concat_step);

    // Create result array
    Mu.Array ret = Mu.zeros(ret_shape, xs[0].dtype);
    unowned float[] ret_data = (float[]) ret.bytes.data;

    // Core function to copy
    ConcatenateNdArray_CopyLowerFunc copy_lower_func = (concat_idx, ret_idx_base0, src_idx_base) => {
        int ret_idx_base1 = ret_idx_base0 + concat_offsets[concat_idx];
        float[] src_data = ((float[]) xs[concat_idx].bytes.data)[xs[concat_idx].start:];
        for (int l_idx = 0; l_idx < n_lowers[concat_idx]; l_idx++) {
            ret_data[ret_idx_base1 + l_idx] = src_data[src_idx_base + l_idx];
        }
    };

    // Switch by upper size
    if (n_upper == 1) {
        // Run in parallel of stack direction
        RunParallel((int)(n_concat), (concat_idx) => {
            copy_lower_func((size_t)(concat_idx), 0, 0);
        });
    } else {
        // Run in parallel of high dimension's direction
        RunParallel(n_upper, (u_idx) => {
            int ret_idx_base0 = u_idx * concat_step;
            for (size_t concat_idx = 0; concat_idx < n_concat; concat_idx++) {
                int src_idx_base = u_idx * n_lowers[concat_idx];
                copy_lower_func(concat_idx, ret_idx_base0, src_idx_base);
            }
        });
    }

    return ret;
}
