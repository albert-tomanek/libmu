namespace Mu
{
	public Array mul(Array a, Array b)
	requires(a.dtype == b.dtype)
	{
		// https://numpy.org/doc/stable/user/basics.broadcasting.html

		/* Compute the shape that the output will have after broadcasting */
		int[] new_shape = new int[int.max(a.shape.length, b.shape.length)];

		for (int dim = 0; dim < new_shape.length; dim++)
		{
			int dim_a = (dim >= new_shape.length - a.shape.length) ? a.shape[dim - (new_shape.length - a.shape.length)] : 1;
			int dim_b = (dim >= new_shape.length - b.shape.length) ? b.shape[dim - (new_shape.length - b.shape.length)] : 1;

			if (dim_a == dim_b || dim_a == 1 || dim_b == 1)
			{
				new_shape[dim] = int.max(dim_a, dim_b);
			}
			else
			{
				error("Shapes %s and %s are not compatible in dimention %d.", print_shape(a.shape), print_shape(b.shape), dim);
			}
		}

		/* Create the result array */
		var result = Mu.Array.zeros(new_shape, a.dtype);

		// int[] idx = new int[result.shape.length];	// {0, 0, 0, ...}
		// flatten_index(int[] index, int[] shape)
		// get_modulo(int[] index)	// index can be > shape

		int[] idx = new int[result.shape.length];		// {0, 0, 0, ...} This is used by mul_recersive to keep track of where it's at.

		mul_recursive(ref idx, 0, result, a, b);

		return result;
	}

	/* This function calls itself recursively for each dimension,
	 * so that all combinations of indices happen. Ie:
	 * (1, 1, 1)
	 * (1, 1, 2)
	 * (1, 1, 3)
	 * (1, 2, 1)      [but starting from 0 ;-)]
	 * ...
	 */
	private void mul_recursive(ref int[] idx, int dim, Array res, Array a, Array b)	// Forget ref-counting
	{
		for (idx[dim] = 0; idx[dim] < res.shape[dim]; idx[dim]++)
		{
			if (dim == idx.length - 1)
			{
				int[] a_idx = idx_remainder(idx, a.shape);
				int[] b_idx = idx_remainder(idx, b.shape);

				int   a_offset = idx_offset(a_idx, a.shape);
				int   b_offset = idx_offset(b_idx, b.shape);
				int   res_offset = idx_offset(idx, res.shape);

				float p = ((float[]) a.bytes.data)[a_offset];
				float q = ((float[]) b.bytes.data)[b_offset];
				((float[]) res.bytes.data)[res_offset] = p * q;

				// message(@"$(print_shape(idx)) ($res_offset) = $(print_shape(a_idx)) * $(print_shape(b_idx))");
			}
			else
			{
				mul_recursive(ref idx, dim + 1, res, a, b);
			}
		}
	}
}
