struct Slice
{
    int first;
    int second;
}

int ClipOp(int val, int lower, int upper)
{
    if (val < lower)
        return lower;
    if (val > upper)
        return upper;
    return val;
}

internal Mu.Array MakeSlice(Mu.Array src, int[,] indices)
{
    Slice si = {0, 0};
    unowned int[] shape = src.shape;
 
    // Compute slice shape and new positive index
    int[] slice_shape = {};
    Slice[] new_index = {};
    for (size_t i = 0; i < shape.length; i++) {
        si.first = indices[i, 0]; si.second = indices[i, 1];
        if (indices.length[0] <= i) {
            // All
            slice_shape += (shape[i]);
            new_index += (Slice() {first= 0, second= shape[i]});
        } else {
            // Make index positive
            int s = (0 <= si.first) ? si.first : shape[i] + si.first;
            int e = (0 <= si.second) ? si.second : shape[i] + si.second;
            // Clamp
            s = ClipOp(s, 0, shape[i]);  // must be next of the last.
            e = ClipOp(e, 0, shape[i]);
            // Register
            slice_shape += (int.max(e - s, 0));  // Escape negative
            new_index += (Slice() {first= s, second= e});
        }
    }
    
    // Copy to slice array
    return CopySlice(src, slice_shape, new_index);
}

internal void CopySliceImpl([CCode(array_length = false)] float[] src_data,
                            [CCode(array_length = false)] float[] ret_data, int[] ret_shape,
                          int[] prev_offsets,
                          int[] post_offsets,
                          int src_step_top, int ret_step_top) {
    size_t n_depth = ret_shape.length;

    // Run in parallel (Only top level)
    RunParallel(ret_shape[0], (ret_top) => {
        int ret_idx_base = ret_top * ret_step_top;
        int src_idx_base = ret_top * src_step_top + prev_offsets[0];

        // Create stacks and counter
        int[] ret_cnts = new int[n_depth];
        size_t depth = 1;  // Start from 1
        int src_idx = 0;

        for (int ret_idx = 0; ret_idx < ret_step_top; ret_idx++) {
            // Go down
            for (; depth < n_depth; depth++) {
                src_idx += prev_offsets[depth];  // Forward prev offset
            }

            // Operate
            ret_data[ret_idx_base + ret_idx] = src_data[src_idx_base + src_idx];
            src_idx += 1;  // Forward normally

            // Go up and count (Down to 1)
            for (; 1 < depth; depth--) {
                size_t prev_d = depth - 1;
                ret_cnts[prev_d]++;  // Count up
                if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                    break;  // Continue normally
                }
                // Go upper depth
                ret_cnts[prev_d] = 0;             // Clear count
                src_idx += post_offsets[prev_d];  // Forward post offset
            }
        }
    });
}

int[] ComputeSliceOffset(int[] child_sizes,
                                    Slice[] slice_index,
                                    int[] src_shape, bool IsPrev) {
    int[] offsets = {};
    for (size_t depth = 0; depth < child_sizes.length; depth++) {
        var si = slice_index[depth];
        int len = (IsPrev ? si.first : src_shape[depth] - si.second);
        offsets += (child_sizes[depth] * len);
    }
    return offsets;
}

internal Mu.Array CopySlice(Mu.Array src, int[] ret_shape,
                         Slice[] slice_index) {
    // Pre-compute child sizes
    int[] child_sizes = ComputeChildSizes(src.shape);
    // Pre-compute offsets
    int[] prev_offsets =
            ComputeSliceOffset(child_sizes, slice_index, src.shape, true);
    int[] post_offsets =
            ComputeSliceOffset(child_sizes, slice_index, src.shape, false);

    // Pre-compute top steps for parallel
    int ret_step_top = 1;
    for (size_t i = 1; i < ret_shape.length; i++)
        ret_step_top *= ret_shape[i];
    int src_step_top = child_sizes[0];

    // Create slice instance
    Mu.Array ret = Mu.zeros(ret_shape, src.dtype);

    // Start to copy
    CopySliceImpl(((float[]) src.bytes.data)[src.start:], ((float[]) ret.bytes.data), ret_shape, prev_offsets, post_offsets,
                  src_step_top, ret_step_top);

    return ret;
}
