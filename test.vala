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
	assert(scalar.value == 1.0);

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
	var a = Mu.arange(1, 7);
	a = a.reshape({2, 3});

	var rep = Mu.repeat(a, 2, 0);
	print(@"$rep\n");

	// Test broadcasting
	{
		float[] data = {
			0.0f,0.0f,1.0f, 0.0f,0.5f,0.5f, 0.5f,1.0f,0.0f,
			0.0f,0.5f,0.5f, 1.0f,1.0f,0.5f, 0.5f,1.0f,0.0f,
			0.5f,0.5f,0.0f, 0.0f,0.5f,0.0f, 0.0f,0.5f,0.5f
		};
		var img = Mu.Array.from(data, {3, 3, 3});

		float[] mask = {0f, 1f, 0f};
		var img_green = Mu.mul(img, Mu.Array.from(mask, {3}));

		print(@"$img_green\n");
		// assert(max(img_green[:,:,0]) == 0)
	}

	// Test more sofisticated broadcasting
	// {
	// 	float[] data_a = {10, 12};	// [[[10],[12]]]
	// 	float[] data_b = {3, 4};	// [[[3]],[[4]]]
	//
	// 	var result = Mu.mul(
	// 		Mu.Array.from(data_a, {1,2,1}),
	// 		Mu.Array.from(data_b, {2,1,1})
	// 	);
	//
	// 	assert(Mu.shape_eq(result.shape, {2,2,1}));
	// 	//assert(result.flatten == {30, 36, 40, 48});
	//
	// 	print(@"$(Mu.print_shape(result.shape))\n");
	// 	print(@"$result\n");
	// }
}
