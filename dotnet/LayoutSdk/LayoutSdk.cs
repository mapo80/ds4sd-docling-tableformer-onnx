using SkiaSharp;
using System.Collections.Concurrent;
using System.IO;
using System.Collections.Generic;

namespace LayoutSdk;

public class LayoutSdk : IDisposable
{
    internal readonly ConcurrentDictionary<LayoutEngine, ILayoutBackend> Backends = new();
    private readonly LayoutSdkOptions _options;

    public LayoutSdk(LayoutSdkOptions options)
    {
        _options = options;
    }

    public LayoutResult Process(string imagePath, bool overlay, LayoutEngine engine)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
            throw new ArgumentException("Image path is empty", nameof(imagePath));
        if (!File.Exists(imagePath))
            throw new FileNotFoundException("Image not found", imagePath);

        using var bitmap = SKBitmap.Decode(imagePath) ?? throw new InvalidOperationException("Unable to decode image");
        var backend = Backends.GetOrAdd(engine, CreateBackend);
        var boxes = backend.Infer(bitmap);
        SKBitmap? overlayImg = null;
        if (overlay)
            overlayImg = DrawOverlay(bitmap, boxes);
        return new LayoutResult(boxes, overlayImg);
    }

    private ILayoutBackend CreateBackend(LayoutEngine engine) => engine switch
    {
        LayoutEngine.Onnx => new OnnxRuntimeBackend(_options.OnnxModelPath),
        LayoutEngine.Ort => new OnnxRuntimeBackend(_options.OrtModelPath),
        LayoutEngine.Openvino => new OpenVinoBackend(_options.OpenVinoModelPath),
        _ => throw new ArgumentOutOfRangeException(nameof(engine), engine, null)
    };

    private static SKBitmap DrawOverlay(SKBitmap baseImage, IReadOnlyList<BoundingBox> boxes)
    {
        var copy = baseImage.Copy();
        using var canvas = new SKCanvas(copy);
        using var paint = new SKPaint { Color = SKColors.Lime, IsStroke = true, StrokeWidth = 2 };
        foreach (var b in boxes)
            canvas.DrawRect(b.X, b.Y, b.Width, b.Height, paint);
        return copy;
    }

    public void Dispose()
    {
        foreach (var backend in Backends.Values)
            if (backend is IDisposable d) d.Dispose();
    }
}

public record LayoutSdkOptions(string OnnxModelPath, string OrtModelPath, string OpenVinoModelPath);
