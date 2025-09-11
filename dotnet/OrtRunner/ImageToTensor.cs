using System;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

public static class ImageToTensor
{
    public static DenseTensor<float> ToCHWFloat(string imagePath, int targetH, int targetW, float[] mean, float[] std)
    {
        using var bmp = SKBitmap.Decode(imagePath);
        if (bmp == null) throw new FileNotFoundException(imagePath);
        using var resized = bmp.Resize(new SKImageInfo(targetW, targetH), SKFilterQuality.High);
        if (resized == null) throw new InvalidOperationException("Resize failed");
        var tensor = new DenseTensor<float>(new[] { 1, 3, targetH, targetW });
        for (int y = 0; y < targetH; y++)
        {
            for (int x = 0; x < targetW; x++)
            {
                var color = resized.GetPixel(x, y);
                float r = color.Red / 255f;
                float g = color.Green / 255f;
                float b = color.Blue / 255f;
                tensor[0, 0, y, x] = (r - mean[0]) / std[0];
                tensor[0, 1, y, x] = (g - mean[1]) / std[1];
                tensor[0, 2, y, x] = (b - mean[2]) / std[2];
            }
        }
        return tensor;
    }
}
