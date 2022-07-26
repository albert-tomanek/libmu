/* Compile: valac *.vala ../shape.vala -o tinyndarray */

template <typename F>
NdArray ApplyDualOp(const NdArray& lhs, const NdArray& rhs, F op) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        NdArray ret(lhs.shape());
        // Simply apply all
        ApplyOpSimple(ret, lhs, rhs, op);
        return ret;
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Apply broadcast
        NdArray ret(ret_shape);
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(op));
        return ret;
    }
}

void main()
{
}