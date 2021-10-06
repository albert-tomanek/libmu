// Used to fit less dimentional arrays onto more dimentional arrays.
internal int[] idx_remainder(int[] idx, int[] shape)
requires(idx.length <= shape.length)
{
	int[] result = new int[shape.length];

	for (int dim = 0; dim < shape.length; dim++)
	{
		if (dim >= shape.length - idx.length)
		{
			// Get the remainder.
			result[dim] = idx[dim] % shape[dim];
		}
		else
		{
			result[dim] = 0;	// You have to account for indexes smaller than the shape. Ie when multiplying an array of shape {3,3,3} by an array of shape {3}, return {0%3, 0%3, 3%3} instead of just {3%3}.
		}
	}

	return result;
}

// @return The offset (in a flat array) of the element at `idx` in an array of `shape`.
internal int idx_offset(int[] idx, int[] shape)
requires(idx.length == shape.length)
{
	int offset = 0;

	for (int dim = 0; dim < shape.length; dim++)
	{
		offset += Mu.shape_length(shape[dim+1:]) * idx[dim];
	}

	return offset;
}
