void main()
{
	float[] data = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15, 16};
	var a = Mu.Array.from(data, {1, 2, 4, 2});
	print(@"$a\n");
}
