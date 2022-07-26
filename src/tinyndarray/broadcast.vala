// ------------------ Utilities for NdArray (Broadcast common) -----------------

static Shape CheckBroadcastable(const Shape& l_shape, const Shape& r_shape) {
    // We assuming left array has deeper shape than right one.
    if (l_shape.size() < r_shape.size()) {
        return CheckBroadcastable(r_shape, l_shape);  // Swap
    }
    // `l_shape.size()` is maximum depth.

    // Check empty
    if (r_shape.size() == 0 || (r_shape.size() == 1 && r_shape[0] == 0)) {
        throw std::runtime_error("Broadcast of empty array");
    }

    // Compute broadcasted shape
    Shape shape(l_shape.size());
    size_t r_offset = l_shape.size() - r_shape.size();
    for (size_t i = 0; i < l_shape.size(); i++) {
        if (i < r_offset) {
            shape[i] = l_shape[i];
        } else {
            const int l = l_shape[i];
            const int r = r_shape[i - r_offset];
            if (l == r) {
                shape[i] = l;  // no broadcast
            } else if (l == 1) {
                shape[i] = r;  // left broadcast
            } else if (r == 1) {
                shape[i] = l;  // right broadcast
            } else {
                std::stringstream ss;
                ss << "Non operatable shape";
                ss << " (" << l_shape << " vs " << r_shape << ")";
                throw std::runtime_error(ss.str());
            }
        }
    }
    return shape;
}

static Shape PadShape(const Shape& shape, size_t size) {
    if (size < shape.size()) {
        throw std::runtime_error("Invalid shape to pad");
    }
    const size_t n_pad = size - shape.size();
    Shape ret_shape;
    ret_shape.reserve(size);
    ret_shape.resize(n_pad, 1);                                     // Fill by 1
    ret_shape.insert(ret_shape.end(), shape.begin(), shape.end());  // Concat
    return ret_shape;
}

static size_t ReduceShapesBroadcast(Shape& ret_shape, Shape& l_shape,
                                    Shape& r_shape, const size_t depth_offset) {
    // Require `ret_shape.size() == l_shape.size() == r_shape.size()`

    // Remove meaningless dimensions.
    Shape ret_shape_cleaned, l_shape_cleaned, r_shape_cleaned;
    int size_pool = 1;
    size_t depth = 0;
    for (; depth < ret_shape.size() - depth_offset; depth++) {
        if (l_shape[depth] == r_shape[depth]) {
            // Store
            size_pool *= l_shape[depth];
        } else {
            // Pop
            if (size_pool != 1) {
                ret_shape_cleaned.push_back(size_pool);
                l_shape_cleaned.push_back(size_pool);
                r_shape_cleaned.push_back(size_pool);
                size_pool = 1;
            }
            // Through current dimension
            ret_shape_cleaned.push_back(ret_shape[depth]);
            l_shape_cleaned.push_back(l_shape[depth]);
            r_shape_cleaned.push_back(r_shape[depth]);
        }
    }
    // Pop
    if (size_pool != 1 || ret_shape_cleaned.size() == 0) {
        ret_shape_cleaned.push_back(size_pool);
        l_shape_cleaned.push_back(size_pool);
        r_shape_cleaned.push_back(size_pool);
    }
    // Store actual depth count
    const size_t n_depth = ret_shape_cleaned.size();
    // Pass through included in `depth_offset`.
    for (; depth < ret_shape.size(); depth++) {
        ret_shape_cleaned.push_back(ret_shape[depth]);
        l_shape_cleaned.push_back(l_shape[depth]);
        r_shape_cleaned.push_back(r_shape[depth]);
    }
    // Return
    ret_shape = std::move(ret_shape_cleaned);
    l_shape = std::move(l_shape_cleaned);
    r_shape = std::move(r_shape_cleaned);
    return n_depth;
}

template <typename F>
void ApplyOpBroadcastImpl(const NdArray::Iter& ret_data,
                          const NdArray::ConstIter& l_data,
                          const NdArray::ConstIter& r_data,
                          const Shape& ret_shape, const int ret_size,
                          const std::vector<int>& l_steps,
                          const std::vector<int>& r_steps,
                          const size_t start_depth, const size_t n_depth,
                          const int ret_step, F op) {
    // Create stacks and counter
    std::vector<int> ret_cnts(n_depth);
    std::vector<int> l_idx_stack(n_depth), r_idx_stack(n_depth);
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
        op(ret_data + ret_idx, l_data + l_idx, r_data + r_idx);

        // Go up and count
        for (; start_depth < depth; depth--) {
            const size_t prev_d = depth - 1;
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

template <typename F>
void ApplyOpBroadcast(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                      const size_t depth_offset, const int ret_step, F op) {
    Shape ret_shape = ret.shape();

    // Pre-compute padded shapes
    Shape l_shape = PadShape(lhs.shape(), ret_shape.size());
    Shape r_shape = PadShape(rhs.shape(), ret_shape.size());

    // Pre-compute reduced shapes
    const size_t n_depth =
            ReduceShapesBroadcast(ret_shape, l_shape, r_shape, depth_offset);

    // Pre-compute child sizes
    const std::vector<int>& ret_child_sizes = ComputeChildSizes(ret_shape);
    const std::vector<int>& l_child_sizes = ComputeChildSizes(l_shape);
    const std::vector<int>& r_child_sizes = ComputeChildSizes(r_shape);

    // Pre-compute steps
    std::vector<int> l_steps, r_steps;
    l_steps.reserve(n_depth);
    r_steps.reserve(n_depth);
    for (size_t depth = 0; depth < n_depth; depth++) {
        const int& l_s = l_shape[depth];
        const int& r_s = r_shape[depth];
        const int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        const int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        l_steps.push_back(l_step);
        r_steps.push_back(r_step);
    }

#if 1  // Run in parallel
    RunParallel(ret_shape[0], [&](int i) {
        const int ret_size = static_cast<int>(ret.size()) / ret_shape[0];
        ApplyOpBroadcastImpl(ret.data() + ret_child_sizes[0] * i,
                             lhs.data() + l_steps[0] * i,
                             rhs.data() + r_steps[0] * i, ret_shape, ret_size,
                             l_steps, r_steps, 1, n_depth, ret_step, op);
    });
#else  // Run sequentially
    ApplyOpBroadcastImpl(ret.data(), lhs.data(), rhs.data(), ret_shape,
                         static_cast<int>(ret.size()), l_steps, r_steps, 0,
                         n_depth, ret_step, op);
#endif
}

template <typename F>
inline auto WrapOpForIter(F op) {
    return [op](const NdArray::Iter& o, const NdArray::ConstIter& l,
                const NdArray::ConstIter& r) {
        *o = op(*l, *r);  // wrap pointer operation for iterator's one
    };
}
