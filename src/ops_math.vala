namespace Mu
{
	/* Equality */
	public bool eq(Array a, Array b)
	requires(a.dtype == b.dtype)
	{
		if (!shape_eq(a.shape, b.shape))
			return false;
		
		if (Memory.cmp(
			(void *) ((float[]) a.bytes.data)[a.start:],
			(void *) ((float[]) b.bytes.data)[b.start:],
			shape_length(a.shape) * dtype_size(a.dtype))
			!= 0
		)
			return false;
		
		return true;
	}

	/* Arithmetic (broadcastable) */

	public Array add(Array a, Array b)
	requires(a.dtype == b.dtype)
	{
		return ApplyDualOp(a, b, (p, q) => p + q);
	}

	public Array sub(Array a, Array b)
	requires(a.dtype == b.dtype)
	{
		return ApplyDualOp(a, b, (p, q) => p - q);
	}

	public Array mul(Array a, Array b)
	requires(a.dtype == b.dtype)
	{
		return ApplyDualOp(a, b, (p, q) => p * q);
	}

	public Array div(Array a, Array b)
	requires(a.dtype == b.dtype)
	{
		return ApplyDualOp(a, b, (p, q) => p / q);
	}

	/* Axis functions */

	public Array sum(Array a, int[]? axes = {}, bool keepdims = false)
	{
		return ReduceAxis(a, axes, keepdims, 0f, (p, q) => p + q);
	}

	public Array mean(Array a, int[]? axes = {}, bool keepdims = false)
	{
		if (Mu.shape_length(a.shape) == 0)
			return scalar(float.NAN);

		var sum = sum(a, axes, keepdims);
		return div(sum, scalar((float)shape_length(a.shape) / (float)shape_length(sum.shape)));
	}

	/* Matrix ops */
	public Array dot(Array a, Array b)
	{
		return DotNdArray(a, b);
	}
}
