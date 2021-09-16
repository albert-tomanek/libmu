namespace Mu
{
	public Array repeat(Array arr, int n_repeats, int axis)
	requires(n_repeats > 0)
	{
		if (axis < 0 || arr.shape.length <= axis)
		{
			error("Axis %d is out of bounds for array of dimension %d.", axis, arr.shape.length);
		}

		/* Calculate the new shape */
		var new_shape = arr.shape;
		new_shape[axis] = arr.shape[axis] * n_repeats;

		var ret = Array.zeros(new_shape, arr.dtype);

		/* Copy the data */

		int copy_len = shape_length(arr.shape[axis+1:]);

		for (int i = 0; i < shape_length(arr.shape[:axis+1]); i++)	// Cycle through each chunk from the source to be copied.
		{
			int src_idx = arr.start + i * copy_len;

			for (int cur_repeat = 0; cur_repeat < n_repeats; cur_repeat++)
			{
				int dst_idx = (i * n_repeats + cur_repeat) * copy_len;		// no need for `start +` because we're copying into a new array.

				message(@"copy $src_idx+$copy_len to $dst_idx+$copy_len");
				copy_items(ret, dst_idx, arr, src_idx, copy_len);
			}
		}

		return ret;
	}

	public Array arange(double start, double stop, double step = 1, DType dtype = DType.FLOAT32)
	requires(step != 0)
	{
		int n_items = (int) (double.max(stop - start, 0) / step);

		var arr = Array.zeros({n_items}, dtype);

		for (int i = 0; i < n_items; i++)
		{
			((float[]) arr.bytes.data)[i] = (float) (start + i * step);
		}

		return arr;
	}
}
