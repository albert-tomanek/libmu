namespace Mu.Utils
{
    public uint8[] to_rgb(Array arr)
    requires(arr.shape.length == 3)
    {
        uint8[] data = new uint8[shape_length(arr.shape)];
        
        for (int i = 0; i < data.length; i++)
            data[i] = (uint8)(((float[]) arr.bytes.data)[arr.start + i] * 255);

        return data;
    }

    public Array from_rgb(uint8[] data, int width, int height)
    {
        var arr = Mu.zeros({height, width, 3});

        for (int i = 0; i < shape_length(arr.shape); i++)
            ((float[]) arr.bytes.data)[i] = (float) data[i];

        return Mu.div(arr, Mu.scalar(255));
    }
}