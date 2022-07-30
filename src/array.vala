namespace Mu
{
	[CCode(cprefix="MU_")]
	public enum DType
	{
		FLOAT32,
	}

	int dtype_size(DType dtype)
	{
		switch (dtype)
		{
			case DType.FLOAT32: return 4;
			default:
				error(@"Unsupported dtype `$(dtype)`");
		}
	}

	public class Array
	{
		public int[] shape { get; internal set; }
		public DType dtype { get; internal set; }

		public int ndim { get { return this.shape.length; } }

		// internal TypeOps type_ops;

		internal ByteArray bytes;
		internal int start;

		internal Array.unalloc(int[] shape, DType dtype)
		{
			// This constructor is for when you allocate your own data later on. If you need an already allocated array to output to, use Mu.zeros()

			this.shape = shape;	// shape is actually 2 arguments under the hood so we can't use Object(shape: ..., dtype: ...) unfortunately. :-(
			this.dtype = dtype;

			this.bytes = new ByteArray();
			this.start = 0;
		}

		internal Array.with_bytes(ByteArray bytes, int start, int[] shape, DType dtype = DType.FLOAT32)
		{
			// This constructor is for when the data was already allocated and you just want this array to share it.

			this.shape = shape;
			this.dtype = dtype;

			this.bytes = bytes;	// Data gets REFERENCED. Changes made to this array will be reflected in the source array as well.
			this.start = start;
		}

		/* Factory methods */

		public static Array from(void *data, int[] shape, bool steal = false, DType dtype = DType.FLOAT32)
		{
			// Create a new array using a copy of `data`.
			var arr = new Array.unalloc(shape, dtype);

			if (steal)
			{
				arr.bytes.set_size(shape_length(shape) * dtype_size(dtype));
				arr.bytes = new ByteArray.take((uint8[]) data);
			}
			else
			{
				size_t size = shape_length(shape) * dtype_size(dtype);
				arr.bytes.set_size((uint) size);
				Memory.copy(arr.bytes.data, data, size);
			}

			return arr;
		}

		/* Duplication */

		public Array copy()
		{
			Array ret = new Array.unalloc(this.shape, this.dtype);
			
			ret.bytes.set_size(shape_length(this.shape) * dtype_size(this.dtype));
			Memory.copy(ret.bytes.data, (void *) this.bytes.data[this.start * dtype_size(this.dtype)], ret.bytes.len);	// If this.start is nonzero, this will only copy the subset of the data that is used by this array.
			
			return ret;
		}

		/* Array access */

		public float value {
			/* Set the value of 0d arrays directly */

			get {
				if (shape_length(this.shape) != 1)
				{
					error("You may only get the literal value of an array with shape {1}. This array has shape %s.", print_shape(this.shape));
				}
				else
				{
					return ((float[]) this.bytes.data)[this.start + 0];
				}
			}
			set {
				if (shape_length(this.shape) != 1)
				{
					error("You may only set the literal value of an array with shape {1}. This array has shape %s.", print_shape(this.shape));
				}
				else
				{
					((float[]) this.bytes.data)[this.start + 0] = value;
				}
			}
		}

		public new void set(Array that)
		requires(that.dtype == this.dtype)
		{
			if (!shape_eq(that.shape, this.shape))
			{
				error("Can't `set` array of shape %s to array of shape %s.", print_shape(this.shape), print_shape(that.shape));
			}

			Memory.copy(
				((uint8 *) this.bytes.data) + (this.start * dtype_size(this.dtype)),
				((uint8 *) that.bytes.data) + (that.start * dtype_size(this.dtype)),
				shape_length(this.shape)*dtype_size(this.dtype)
			);
		}

		[CCode(sentinel = "G_MININT")]	// Couldn't think of a better sentinel if we're gonna support negative indices...
		public new Array get(int i0, ...)
		{
			// You don't want to use this function in a loop because it has to create & return a new GObject every time it is called.

			int[] indices = _parse_indices(i0, va_list());

			return this.get_i(indices);
		}

		public Array get_i(int[] indices)
		{
			if (indices.length > this.shape.length)
			{
				error("Too many indices for array: array is %d-dimensional, but %d were indexed.", this.shape.length, indices.length);
			}

			/* Find the correct start position */
			int start = this.start;

			for (int i = 0; i < indices.length; i++)
			{
				int positive_index = 0 <= indices[i] ? indices[i] : indices[i] + this.shape[i];

				if (positive_index < 0 || positive_index >= this.shape[i])
				{
					error("Index %d is out of bounds for axis %d with size %d.", indices[i], i, this.shape[i]);
				}

				start += positive_index * shape_length(this.shape[i+1:]);
			}

			return new Array.with_bytes(this.bytes, start, this.shape[indices.length:], this.dtype);
		}

		public long length { get { return this.shape[0]; } }	// Vala requires this to be able to leave out the end idx in `array[3:]`
		public Array slice(long start, long end)
		{
			if (start < -this.shape[0] || start > this.shape[0] ||
				end < -this.shape[0] || end > this.shape[0])
			{
				error("Slice [%s:%s] is out of bounds for array of shape %s.", start != 0 ? start.to_string() : "", end != 0 ? end.to_string() : "", print_shape(this.shape));
			}

			if (start < 0) start += this.shape[0];
			if (end < 0) end += this.shape[0];

			if (start > end)
			{
				error("Slice end index %ld is before start index %ld.", end, start);
			}

			var before_start = this.shape;
			before_start[0] = (int) start;
			var shape = this.shape;
			shape[0] = (int) (end - start + 1);

			return new Array.with_bytes(this.bytes, shape_length(before_start), shape, this.dtype);
		}
		
		public Array deep_slice(int[,] indices)	// NOTE: Unlike in numpy, this creates a copy of the data.
		{
			if (indices.length[1] != 2)
				error("Slicing requires pairs of indices, but provided array has shape {%d, %d}. (must end in 2)", indices.length[0], indices.length[1]);
			
			return MakeSlice(this, indices);
		}

		/* Iteration */

		public delegate void ForeachCb(Array sub, int[] idx);

		public void @foreach(ForeachCb callback, int max_depth = -1)
		requires(max_depth <= this.shape.length)
		requires(max_depth >= 0 || max_depth == -1)
		{
			if (max_depth == 0)
			{
				callback(this, {});
			}
			else
			{
				int[] idx = new int[this.shape.length];	// {0, 0, ...}
				foreach_rec(this, idx, 1, (max_depth == -1) ? int.MAX : max_depth, callback);
			}
		}

		static void foreach_rec(Array arr, /*out*/ int[] idx, int dim, int max_dim, ForeachCb callback)
		{
			for (idx[dim-1] = 0; idx[dim-1] < arr.shape[dim-1]; idx[dim-1]++)
			{
				if (dim == arr.shape.length || dim == max_dim)
				{
					callback(arr.get_i(idx[:dim]), idx[:dim]);
				}
				else
				{
					foreach_rec(arr, idx, dim + 1, max_dim, callback);
				}
			}
		}

		public delegate float ApplyFn(float val);

		public Array apply(ApplyFn fn)
		{
			var result = this.copy();
			unowned float[] data = (float[]) result.bytes.data;

			for (int i = 0; i < shape_length(result.shape); i++)
			{
				data[i] = fn(data[i]);
			}

			return result;
		}

		/* Mutating arrays */

		public Array reshape(int[] new_shape)
		{
			if (shape_length(this.shape) != shape_length(new_shape))
			{
				error("Cannot reshape array of shape %s into shape %s.", print_shape(this.shape), print_shape(new_shape));
			}

			return new Array.with_bytes(this.bytes, this.start, new_shape, this.dtype);
		}

		/* Convenience things */

		public string to_string()
		{
			var repr = dim_printer(this.bytes.data, this.start, this.shape, this.shape.length);

			return @"Mu.Array($repr)";
		}

		private static int PRINT_LIMIT = 20;

		private static string dim_printer(uint8[] data, int start, int[] shape, int dim)
		{
			int dim_len = (dim == 0) ? 1 : shape[shape.length - dim];	// If it's a scalar (ie. zero dimentions), treat it as a 1-length 1-dimentional array.

			string str = "";

			int limit = int.min(dim_len, PRINT_LIMIT + 1);
			for (int i = 0; i < limit; i++)
			{
				if (dim == 1 || dim == 0)
				{
					str += (i == PRINT_LIMIT) ? "..." : "%f".printf(((float[]) data)[start + i]);
					str += (i == limit - 1) ? "" : ", ";
				}
				else if (dim > 1)
				{
					int skip = shape_length(shape[shape.length - dim + 1:]); 	// the number of items you need to skip to get to the next item in this dimension.

					str += (i != 0) ? string.nfill((shape.length - dim) * 1, ' ') : "";
					if (i == PRINT_LIMIT)
						str += "...";
					else
					{
						str += "[";
						str += "%s]".printf(dim_printer(data, start + i*skip, shape, dim - 1));
					}
					str += (i == limit - 1) ? "" : (dim > 2) ? ",\n\n         " : ",\n         ";
				}
			}

			return str;
		}

	}

	/* Creation */

	public Array scalar(float val)
	{
		float[] data = {val};
		return Array.from((owned) data, {1}, true);
	}
	
	public Array array(float[] data, int[] shape)
	{
		if (shape_length(shape) != data.length)
			error("Shape %s doesn't account for all %d elements of array provided.", print_shape(shape), data.length);

		return Array.from(data, shape, false);
	}

	public Array zeros(int[] shape, DType dtype = DType.FLOAT32)
	{
		var arr = new Array.unalloc(shape, dtype);
		arr.bytes.set_size(shape_length(shape) * dtype_size(dtype));	// Should set the 'length' as well.

		for (int i = 0; i < shape_length(shape); i++)
		{
			((float[]) arr.bytes.data)[i] = 0.0f;
		}

		return arr;
	}

	public Array ones(int[] shape, DType dtype = DType.FLOAT32)
	{
		var arr = new Array.unalloc(shape, dtype);
		arr.bytes.set_size(shape_length(shape) * dtype_size(dtype));	// Should set the 'length' as well.

		for (int i = 0; i < shape_length(shape); i++)
		{
			((float[]) arr.bytes.data)[i] = 1.0f;
		}

		return arr;
	}

	internal void copy_items(Array dst, int dst_index, Array src, int src_index, int n_items)
	requires(dst.dtype == src.dtype)
	{
		Memory.copy(
			((uint8 *) dst.bytes.data) + (dst_index * dtype_size(src.dtype)),
			((uint8 *) src.bytes.data) + (src_index * dtype_size(src.dtype)),
			n_items * dtype_size(src.dtype)
		);
	}

	private int[] _parse_indices(int idx0, va_list rest)
	{
		/* List indices are passed as an int and a va_list (because of how C works).
		 * This function converts them into an int[].									*/

		int[] arr = {};
		arr += idx0;

		while (true) {
			int idx = rest.arg();
			if (idx == int.MIN) break;	// You need to set the `sentinel` ccode so that Vala knows how to terminate the arg list properly.
			arr += idx;
		}

		return arr;
	}
}
