void main()
{
	float[] data = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15, 16};
	int g = 4;
	var a = Mu.Array.from(data, {1, 2, g, 2});
	print(@"$a\n");

	var z = a[7, 3, g];
	print(@"$z");
}
