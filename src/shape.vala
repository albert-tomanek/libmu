namespace Mu
{
	public void shape_valid(int[] shape)
	{
		for (int i = shape.length; 0 < i; i--)
		{
			if (shape[i - 1] <= 0)
			{
				error("Array shape can't have a negative-sized dimension. (Dimension %d has size `%d`).", i - 1, shape[i - 1]);
			}
		}
	}

	public bool shape_eq(int[] shape_a, int[] shape_b)
	{
		if (shape_a.length != shape_b.length) return false;

		for (int i = 0; i < shape_a.length; i++)
		{
			if (shape_a[i] != shape_b[i])
				return false;
		}

		return true;
	}

	public int shape_length(int[] shape)
	{
		if (shape.length == 0) return 1;

		int n = 1;

		foreach (int dim in shape)
		{
			n *= dim;
		}

		return n;
	}

	public string print_shape(int[] shape)
	{
		string list = "";

		for (int i = 0; i < shape.length; i++)
		{
			list += i.to_string();
			list += (i != shape.length - 1) ? ", " : "";
		}

		return @"{$list}";
	}
}
