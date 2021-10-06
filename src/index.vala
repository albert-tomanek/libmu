// Used to fit less dimentional arrays onto more dimentional arrays.
internal int[] idx_remainder(int[] idx, int[] shape)
{
	int[] result = new int[int.min(idx.length, shape.length)];

	for (int dim = -1; dim >= -result.length; dim--)	// Align both arrays by their ends and not their beginnings.
	{
		// Get the remainder.
		result[result.length + dim] = idx[idx.length + dim] % shape[shape.length + dim];	// Remember that 'adding' dim is actually adding -1 or -2  to the length. (offset from *end* of both.)
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
