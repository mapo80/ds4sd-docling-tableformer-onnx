using System;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

public static class ImageToTensor
{
    public static DenseTensor<float> ToCHWFloat(string imagePath, int targetH, int targetW)
    {
        using var bmp = SKBitmap.Decode(imagePath);
        if (bmp == null) throw new FileNotFoundException(imagePath);
        float scale = Math.Min((float)targetW / bmp.Width, (float)targetH / bmp.Height);
        int newW = (int)(bmp.Width * scale);
        int newH = (int)(bmp.Height * scale);
        using var resized = bmp.Resize(new SKImageInfo(newW, newH), SKFilterQuality.High);
        using var canvasBmp = new SKBitmap(targetW, targetH);
        using (var canvas = new SKCanvas(canvasBmp))
        {
            canvas.Clear(SKColors.Black);
            canvas.DrawBitmap(resized, (targetW - newW) / 2f, (targetH - newH) / 2f);
        }
        var tensor = new DenseTensor<float>(new[] { 1, 3, targetH, targetW });
        for (int y = 0; y < targetH; y++)
        {
            for (int x = 0; x < targetW; x++)
            {
                var color = canvasBmp.GetPixel(x, y);
                tensor[0, 0, y, x] = color.Red / 255f;
                tensor[0, 1, y, x] = color.Green / 255f;
                tensor[0, 2, y, x] = color.Blue / 255f;
            }
        }
        return tensor;
    }
}
