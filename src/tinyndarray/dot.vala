internal Mu.Array DotNdArray1d(Mu.Array lhs, Mu.Array rhs) {
    size_t size = Mu.shape_length(lhs.shape);
    if (size != Mu.shape_length(rhs.shape)) {
        error("Invalid size for inner product of 1D");
    }
    // Inner product of vectors
    float[] l_data = ((float[]) lhs.bytes.data)[lhs.start:];
    float[] r_data = ((float[]) rhs.bytes.data)[rhs.start:];
    RunParallelWithReduceFn op = (i) => l_data[i] * r_data[i];
    // Run in parallel
    float sum = RunParallelWithReduce((int)(size), op, (p, q) => p + q, 0);
    return Mu.scalar(sum);
}

delegate void F1d2d(float[] ret_data, float[] l_data, float[] r_data, int n_col, int n_contract);

internal void DotNdArray1d2dImplColMajor([CCode(array_length = false)] float[] ret_data,
                                       [CCode(array_length = false)] float[] l_data,
                                       [CCode(array_length = false)] float[] r_data,
                                       int n_col, int n_contract) {
    // Zero initialization (no parallel)
    Memory.set(ret_data, 0, n_col * sizeof(float));
    // Col-major dot product
    int r_idx = 0;
    for (int l_idx = 0; l_idx < n_contract; l_idx++) {
        for (int col_cnt = 0; col_cnt < n_col; col_cnt++) {
            ret_data[col_cnt] += l_data[l_idx] * r_data[r_idx];
            r_idx++;
        }
    }
}

internal void DotNdArray1d2dImplRowMajor([CCode(array_length = false)] float[] ret_data,
                                         [CCode(array_length = false)] float[] l_data,
                                         [CCode(array_length = false)] float[] r_data,
                                       int n_col, int n_contract) {
    // Row-major dot product
    for (int col_cnt = 0; col_cnt < n_col; col_cnt++) {
        float sum = 0;
        int r_idx = col_cnt;
        for (int l_idx = 0; l_idx < n_contract; l_idx++) {
            sum += l_data[l_idx] * r_data[r_idx];
            r_idx += n_col;
        }
        ret_data[col_cnt] = sum;
    }
}

const int DOT_CACHE_SCALE = 10;

internal F1d2d SelectDot1d2dOp(int[] l_shape, int[] r_shape) {
//      // Debug macros
//  #ifdef TINYNDARRAY_FORCE_DOT_COLMAJOR
//      (void)l_shape;
//      (void)r_shape;
//      return DotNdArray1d2dImplColMajor;  // Col
//  #endif
//  #ifdef TINYNDARRAY_FORCE_DOT_RAWMAJOR
//      (void)l_shape;
//      (void)r_shape;
//      return DotNdArray1d2dImplRowMajor;  // Row
//  #endif

    // Decide which major is better
    int left = l_shape[l_shape.length - 1];
    int right = r_shape[r_shape.length - 2] * r_shape[r_shape.length - 1];
    if (left * DOT_CACHE_SCALE < right) {
        return DotNdArray1d2dImplColMajor;  // Col
    } else {
        return DotNdArray1d2dImplRowMajor;  // Row
    }
}

void DotNdArrayNdMdImpl([CCode(array_length = false)] float[] ret_data,
                        [CCode(array_length = false)] float[] l_data,
                        [CCode(array_length = false)] float[] r_data, int n_l,
                        int n_r, int ret_step, int l_step,
                        int r_step, F1d2d op_1d2d) {
    int n_contract = l_step;
    int n_col = ret_step;
    int ret_idx_base = n_r;
#if 0  // Run in parallel
    if (n_l < n_r) {
        RunParallel(n_r, [&](int r_cnt) {  // Right-hand side loop
            int ret_step_base = ret_idx_base * ret_step;
            int r_idx = r_cnt * r_step;
            int l_idx = 0;
            int ret_idx = r_cnt * ret_step;
            for (int l_cnt = 0; l_cnt < n_l; l_cnt++) {
                op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx,
                        n_col, n_contract);
                l_idx += l_step;
                ret_idx += ret_step_base;
            }
        });
    } else {
        RunParallel(n_l, [&](int l_cnt) {  // Left-hand side loop
            int l_idx = l_cnt * l_step;
            int r_idx = 0;
            int ret_idx = l_cnt * ret_idx_base * ret_step;
            for (int r_cnt = 0; r_cnt < n_r; r_cnt++) {
                op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx,
                        n_col, n_contract);
                r_idx += r_step;
                ret_idx += ret_step;
            }
        });
    }
#else  // Run sequentially
    int l_idx = 0;
    int ret_idx0 = 0;
    for (int l_cnt = 0; l_cnt < n_l; l_cnt++) {
        int r_idx = 0;
        int ret_idx = ret_idx0 * ret_step;
        for (int r_cnt = 0; r_cnt < n_r; r_cnt++) {
            op_1d2d(ret_data[ret_idx:], l_data[l_idx:], r_data[r_idx:], n_col,
                    n_contract);
            r_idx += r_step;
            ret_idx += ret_step;
        }
        l_idx += l_step;
        ret_idx0 += ret_idx_base;
    }
#endif
}

int[] _append(owned int[] a, int[] b)
{
    var a_length_orig = a.length;
    a.resize(a.length + b.length);
    Memory.copy((void *) a[a_length_orig:], (void *) b, b.length * sizeof(int));
    return a;
}

internal Mu.Array DotNdArrayNdMd(Mu.Array lhs, Mu.Array rhs) {
    int[] l_shape = lhs.shape;  // 1 <= l.size
    int[] r_shape = rhs.shape;  // 2 <= r.size

    // The last axis of left and the second-to-last axis of right must be same.
    int n_contract = l_shape[l_shape.length - 1];
    if (n_contract != r_shape[r_shape.length - 2]) {
        error("Dot product can't be done for shapes %s and %s. Dimension -1 of left and -2 of right must be same.", Mu.print_shape(l_shape), Mu.print_shape(r_shape));
    }

    // Result shape
    int[] ret_shape = l_shape[:l_shape.length - 1];
    ret_shape = _append(ret_shape, r_shape[:r_shape.length - 2]);
    ret_shape += (r_shape[r_shape.length - 1]);
    // Result array
    Mu.Array ret = Mu.zeros(ret_shape);

    // Compute 2D shapes and steps
    //   [2, 3, (4)] [5, 6, (4), 7] -> [2, 3, 5, 6, 7]
    int ret_step = r_shape[-1];    // [2, 3, 5, 6, <7>]
    int l_step = n_contract;             // [2, 3, <4>]
    int r_step = n_contract * ret_step;  // [5, 6, <4>, <7>]

    int n_l = (int)(Mu.shape_length(lhs.shape)) / l_step;
    int n_r = (int)(Mu.shape_length(rhs.shape)) / r_step;  // [<5>, <6>, 4, 7]

    // Dot product
    DotNdArrayNdMdImpl((float[]) ret.bytes.data, (float[]) lhs.bytes.data[lhs.start:], (float[]) rhs.bytes.data[rhs.start:], n_l, n_r, ret_step,
                       l_step, r_step, SelectDot1d2dOp(l_shape, r_shape));

    return ret;
}

internal Mu.Array DotNdArray(Mu.Array lhs, Mu.Array rhs) {
    int[] l_shape = lhs.shape;
    int[] r_shape = rhs.shape;
    if (Mu.shape_length(lhs.shape) == 0 || Mu.shape_length(rhs.shape) == 0) {
        // Empty array
        error("Dot product of empty array.");
    } else if (Mu.shape_length(lhs.shape) == 1 || Mu.shape_length(rhs.shape) == 1) {
        // Simple multiply
        return Mu.mul(lhs, rhs);
    } else if (l_shape.length == 1 && r_shape.length == 1) {
        // Inner product of vector (1D, 1D)
        return DotNdArray1d(lhs, rhs);
    } else if (r_shape.length == 1) {
        // Broadcast right 1D array
        int[] shape = l_shape[:l_shape.length - 1];
        return DotNdArrayNdMd(lhs, rhs.reshape({r_shape[0], 1})).reshape(shape);
    } else {
        // Basic matrix product
        return DotNdArrayNdMd(lhs, rhs);
    }
}
