using SkiaSharp;
using System;
using System.Collections.Generic;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal sealed class TableFormerOrtBackend : ITableFormerBackend, IDisposable
{
    private readonly TableFormerOnnxBackend _inner;

    public TableFormerOrtBackend(string modelPath)
        : this(new TableFormerOnnxBackend(InferenceSessionAdapter.CreateOrt(modelPath)))
    {
    }

    internal TableFormerOrtBackend(TableFormerOnnxBackend inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
        => _inner.Infer(image, sourcePath);

    public void Dispose() => _inner.Dispose();
}
