/* Compile: valac -g *.vala ../shape.vala ../array.vala -o tinyndarray */

Mu.Array add(Mu.Array a, Mu.Array b)
{
    return ApplyDualOp(a, b, (p, q) => {
        return p + q;
    });
}

void main()
{
    float[] data_a = {1,2, 3,4, 5,6};
    var a = Mu.Array.from(data_a, {3, 2});

    float[] data_b = {100, 200};
    var b = Mu.scalar(50);//Array.from(data_b, {2});

    var c = add(a, b);

    print(@"$a\n\n$b\n\n$c");
}