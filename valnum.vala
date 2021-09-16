/* Compile: cpp valnum.vala -P -o valnum.i.vala; valac valnum.i.vala -o valnum */

namespace vn
{
	public struct Slice
	{
		int z;
	}

	public class Array<T> : Object
	{
		public uint[] shape { get; internal set; }
	//	public Type   dtype { get; internal set; }
		protected int item_size;
		protected bool is_all;
		protected Slice slice;

		public ByteArray data;

		internal Array.with_data(ByteArray data, uint[] shape)
		{
			assert(data.data.length == ShapeOps.product(shape) * sizeof(T));

			this.data  = data;
			this.shape = shape;
		}

		// public Array<T>(owned T[] data)
		// {
		// }

		construct {
			this.item_size = (int) typeof(T).get_qdata(Quark.try_string("vt-size"));
			message("item size: %d", this.item_size);
		}

		internal delegate void ForeachFunc<T>(out T val);

		internal void @foreach(ForeachFunc func)
		{
			uint8 *cur = (uint8 *) this.data.data;

			for (int i = 0; i < this.data.data.length / sizeof(T); i++)
			{
				func(out cur);
				cur += sizeof(T);
			}
		}

		public /*static*/ void add(Array<T> b)
		{
			Type array_type = typeof(T);
			bool type_chosen = false;

			#define CASE(C)					\
			if (array_type == type_##C)		\
			{								\
				C *a_cur = (C *) this.data.data;	\
				C *b_cur = (C *) b.data.data;		\
													\
				for (int i = 0; i < b.data.data.length / sizeof(C); i++)	\
				{						\
					*a_cur += *b_cur;	\
					a_cur++; b_cur++;	\
				}						\
										\
				type_chosen = true;		\
			}

			CASE(int8)
			CASE(uint8)
			CASE(int16)
			CASE(uint16)
			CASE(int32)
			CASE(uint32)
			CASE(int64)
			CASE(uint64)
			CASE(float)
			CASE(double)
			#undef CASE

			if (!type_chosen)
			{
				error("Operation `add` not supported for type `%s`.", array_type.name());
			}
		}

		/* These are not offered by GLib for some reason */

		static Type type_int8 = typeof(int8);
		static Type type_uint8 = typeof(uint8);
		static Type type_int16 = typeof(int16);
		static Type type_uint16 = typeof(uint16);
		static Type type_int32 = typeof(int32);
		static Type type_uint32 = typeof(uint32);
		static Type type_int64 = typeof(int64);
		static Type type_uint64 = typeof(uint64);
		static Type type_float = typeof(float);
		static Type type_double = typeof(double);
	}

	public Array zeros<T>(uint[] shape)
	{
		uint data_len = ShapeOps.product(shape) * (uint) sizeof(T);
		var  data = new ByteArray.sized(data_len) { len = data_len };	// Have to set len so that it actually thinks its filled
		var  arr  = new Array<T>.with_data(data, shape);

		return arr;
	}

	public Array ones<T>(uint[] shape)
	{
		var arr = vn.zeros<T>(shape);

		arr.foreach((out val) => {
			message("hey");
			val = 1;
		});

		return arr;
	}

	namespace ShapeOps {
		uint product(uint[] shape)
		{
			uint ret = 1;

			foreach (uint n in shape)
				ret *= n;

			return ret;
		}
	}
}

namespace vt
{
	/* GLib is retarded and sees both an int16 and an
	 * int32 as G_TYPE_INT, so we have to define our own.
	 */

	public class base_ {
	}

	#define CASE(C)						\
	public class C : base_ {			\
		static construct {				\
			int size = (int) sizeof(global::C);	\
			typeof(C).set_qdata(Quark.try_string("vt-size"), (void*) size);		\
		}								\
	}

	CASE(int8)
	CASE(uint8)
	CASE(int16)
	CASE(uint16)
	CASE(int32)
	CASE(uint32)
	CASE(int64)
	CASE(uint64)
	CASE(float)
	CASE(double)
	#undef CASE
}

void main ()
{
	message("%d", (int) Quark.from_string("vt-size"));
	message("%p", typeof(vt.int16).get_qdata(Quark.from_string("vt-size")));
	var a = vn.ones<vt.int16>({4});
	message(@"$(a.data.data[0])");
	var z = sizeof(bool);
}
