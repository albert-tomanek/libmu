namespace Mu
{
	public Array mul(Array a, Array b)
	requires(a.dtype == b.dtype)
	{
		for (int dim = -1; dim >= -int.min(a.shape.length, b.shape.length); dim--)	// Start iterating from the last dimention and iterate until the earliest common one.
		{
			if (!(a.shape[dim + a.shape.length] == b.shape[dim + b.shape.length]) &&
				!(a.shape[dim + a.shape.length] == 1) &&
				!(b.shape[dim + b.shape.length] == 1)
			)
			{
				error("Shapes %s and %s are not compatible in dimention %d.", print_shape(a.shape), print_shape(b.shape), dim);
			}
			message("Shapes are compatible %s %s", print_shape(a.shape), print_shape(b.shape));
		}

		Array smaller, bigger;

		if (a.shape.length < b.shape.length)
		{
			smaller = a;
			bigger  = b;
		}
		else
		{
			smaller = b;
			bigger  = a;
		}

		/*  */

		var result = Mu.Array.zeros(bigger.shape, bigger.dtype);

		for (int i = 0; i < shape_length(bigger.shape); i++)
		{
			float p = ((float[]) bigger.bytes.data)[i % shape_length(bigger.shape)];
			float q = ((float[]) smaller.bytes.data)[i % shape_length(smaller.shape)];
			((float[]) result.bytes.data)[i] = p * q;
		}
		// https://numpy.org/doc/stable/user/basics.broadcasting.html

		return result;
	}
}
