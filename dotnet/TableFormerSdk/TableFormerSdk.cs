using SkiaSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;

namespace TableFormerSdk;

public sealed class TableFormerSdk : IDisposable
{
    private readonly ConcurrentDictionary<BackendKey, ITableFormerBackend> _backends = new();
    private readonly TableFormerSdkOptions _options;

    public TableFormerSdk(TableFormerSdkOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public TableStructureResult Process(
        string imagePath,
        bool overlay,
        TableFormerRuntime runtime,
        TableFormerModelVariant variant)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
            throw new ArgumentException("Image path is empty", nameof(imagePath));
        if (!File.Exists(imagePath))
            throw new FileNotFoundException("Image not found", imagePath);

        using var bitmap = SKBitmap.Decode(imagePath) ?? throw new InvalidOperationException("Unable to decode image");
        var backend = _backends.GetOrAdd(new BackendKey(runtime, variant), CreateBackend);
        var boxes = backend.Infer(bitmap, imagePath);
        SKBitmap? overlayImg = null;
        if (overlay)
            overlayImg = DrawOverlay(bitmap, boxes);
        return new TableStructureResult(boxes, overlayImg);
    }

    private ITableFormerBackend CreateBackend(BackendKey key)
    {
        return key.Runtime switch
        {
            TableFormerRuntime.Onnx => new TableFormerOnnxBackend(_options.Onnx.GetModelPath(key.Variant)),
            TableFormerRuntime.Ort => throw new NotSupportedException("ORT backend not yet implemented"),
            TableFormerRuntime.OpenVino => throw new NotSupportedException("OpenVINO backend not yet implemented"),
            _ => throw new ArgumentOutOfRangeException(nameof(key.Runtime), key.Runtime, null)
        };
    }

    private static SKBitmap DrawOverlay(SKBitmap baseImage, IReadOnlyList<TableRegion> boxes)
    {
        var copy = baseImage.Copy();
        using var canvas = new SKCanvas(copy);
        using var paint = new SKPaint { Color = SKColors.Lime, IsStroke = true, StrokeWidth = 2 };
        foreach (var region in boxes)
            canvas.DrawRect(region.X, region.Y, region.Width, region.Height, paint);
        return copy;
    }

    public void Dispose()
    {
        foreach (var backend in _backends.Values)
            if (backend is IDisposable disposable)
                disposable.Dispose();
    }

    public void RegisterBackend(TableFormerRuntime runtime, TableFormerModelVariant variant, ITableFormerBackend backend)
    {
        if (backend is null)
            throw new ArgumentNullException(nameof(backend));
        _backends[new BackendKey(runtime, variant)] = backend;
    }

    private readonly record struct BackendKey(TableFormerRuntime Runtime, TableFormerModelVariant Variant);
}

public record TableFormerSdkOptions(TableFormerModelPaths Onnx);

public record TableFormerModelPaths(string FastModelPath, string? AccurateModelPath)
{
    public string GetModelPath(TableFormerModelVariant variant) => variant switch
    {
        TableFormerModelVariant.Fast => FastModelPath,
        TableFormerModelVariant.Accurate when AccurateModelPath is not null => AccurateModelPath,
        TableFormerModelVariant.Accurate =>
            throw new InvalidOperationException("Accurate model path is not configured"),
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, null)
    };
}
