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

	public class Array : Object
	{
		public int[] shape { get; internal set; }
		public DType dtype { get; internal set; }

		internal ByteArray bytes;
		internal int start;

		internal Array(int[] shape, DType dtype)
		{
			// This constructor is for when you allocate your own data later on

			this.shape = shape;	// shape is actually 2 arguments under the hood so we can't use Object(shape: ..., dtype: ...) unfortunately. :-(
			this.dtype = dtype;

			this.bytes = new ByteArray();
			this.start = 0;
		}

		internal Array.with_bytes(ByteArray bytes, int start, int[] shape, DType dtype = DType.FLOAT32)
		{
			// This constructor is for when the data was already allocated and you jist want this array to use it as well.

			this.shape = shape;
			this.dtype = dtype;

			this.bytes = bytes;
			this.start = start;
		}

		/* Factory methods */

		public static Array from(void *data, int[] shape, DType dtype = DType.FLOAT32)
		{
			// Create a new array using a copy of `data`.
			var arr = new Array(shape, dtype);

			size_t size = shape_length(shape) * dtype_size(dtype);
			arr.bytes.set_size((uint) size);
			Memory.copy(arr.bytes.data, data, size);

			return arr;
		}

		public static Array zeros(int[] shape, DType dtype = DType.FLOAT32)
		{
			var arr = new Array(shape, dtype);
			arr.bytes.set_size(shape_length(shape) * dtype_size(dtype));	// Should set the 'length' as well.

			return arr;
		}

		public static Array ones(int[] shape, DType dtype = DType.FLOAT32)
		{
			var arr = Array.zeros(shape, dtype);

			for (int i = 0; i < shape_length(shape); i++)
			{
				((float[]) arr.bytes.data)[i] = 1.0f;
			}

			return arr;
		}

		/* Duplication */

		/* Array access */

		[CCode(sentinel = "G_MININT")]	// Couldn't think of a better sentinel if we're gonna support negative indices...<
		public Array get(int i0, ...)
		{
			int[] indices = _parse_indices(i0, va_list());

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

		static int[] _parse_indices(int idx0, va_list rest)
		{
			/* List indices are passed as an int and a va_list (because of how C works).
			 * This function converts them into an int[].									*/

			var arr = new GLib.Array<int>();
			arr.append_val(idx0);

			while (true) {
				int idx = rest.arg();
				if (idx == int.MIN) break;	// You need to set the `sentinel` ccode so that Vala knows how to terminate the arg list properly.
				arr.append_val(idx);
			}

			return arr.data;
		}

		public float value {
			get {
				if (this.shape.length != 0)
				{
					error("You may only get the literal value of a 0-dimentional array. This array has %d dimensions (shape %s).", this.shape.length, print_shape(this.shape));
				}
				else
				{
					return ((float[]) this.bytes.data)[this.start + 0];
				}
			}
		}

		/* Convenience things */
		// delegate string DepthIter(uint8[] data, int start_idx, int[] shape, int dimension);

		public string to_string()
		{
			var repr = dim_printer(this.bytes.data, this.start, this.shape, this.shape.length);

			return @"Mu.Array($repr)";
		}

		private static int PRINT_LIMIT = 20;

		private static string dim_printer(uint8[] data, int start, int[] shape, int dim)
		{
			// message(@"dim $dim, shape.length $(shape.length)");
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
}
