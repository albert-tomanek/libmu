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
		public int[] shape { get; private set; }
		public DType dtype { get; private set; }

		private ByteArray bytes = new ByteArray();

		private Array(int[] shape, DType dtype)
		{
			this.shape = shape;	// shape is actually 2 arguments under the hood so we can't use Object(shape: ..., dtype: ...) unfortunately. :-(
			this.dtype = dtype;
		}

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

		// delegate string DepthIter(uint8[] data, int start_idx, int[] shape, int dimension);

		public string to_string()
		{
			var repr = dim_printer(this.bytes.data, 0, this.shape, this.shape.length);

			return @"Mu.Array($repr)";
		}

		private static int PRINT_LIMIT = 20;

		private static string dim_printer(uint8[] data, int start, int[] shape, int dim)
		{
			// message(@"dim $dim, shape.length $(shape.length)");
			int dim_len = shape[shape.length - dim];

			string str = "";

			for (int i = 0; i < dim_len; i++)
			{
				if (dim == 1)
				{
					str += "%f".printf(((float[]) data)[start + i]);
					str += (i == dim_len - 1) ? "" : ", ";
				}
				else if (dim > 1)
				{
					int skip = shape_length(shape[shape.length - dim + 1:]); 	// the number of items you need to skip to get to the next item in this dimension.

					str += (i != 0) ? string.nfill((shape.length - dim) * 1, ' ') : "";
					str += "[";
					str += "%s]".printf(dim_printer(data, start + i*skip, shape, dim - 1));
					str += (i == dim_len - 1) ? "" : (dim > 2) ? ",\n\n         " : ",\n         ";
				}
			}

			return str;
		}

	}
}
