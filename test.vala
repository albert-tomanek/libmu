void main()
{
	test_ops();
}

void test_array()
{
	// Test if printing and `from` works
	float[] data = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15, 16};
	int g = 4;
	var a = Mu.Array.from(data, {2, g, 2});
	print(@"$a\n");

	// Test printing scalars
	var scalar = Mu.Array.from(data, {});
	print(@"$scalar\n");

	// Test slices
	Mu.Array piece = a[1, -1];
	print(@"$piece\n");

	a[1, 3, 1].value = 100f;
	assert(a[1, -1, -1].value == 100f);

	// Test reshaping
	var reshaped = a.reshape({16, 1});
	print(@"$reshaped\n");
}

void test_ops()
{
	float[] data = {1,2,3, 4,5,6};
	var a = Mu.Array.from(data, {2, 3});

	var rep = Mu.repeat(a, 2, 0);
	print(@"$rep\n");
}
