namespace Mu
{
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
}
