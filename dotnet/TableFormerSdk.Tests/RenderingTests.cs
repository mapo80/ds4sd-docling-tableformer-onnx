using SkiaSharp;
using System;
using System.Collections.Generic;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;
using TableFormerSdk.Rendering;
using Xunit;

namespace TableFormerSdk.Tests;

public class RenderingTests
{
    [Fact]
    public void OverlayRenderer_NoRegions_ReturnsNull()
    {
        using var bitmap = new SKBitmap(10, 10);
        var renderer = new OverlayRenderer(TableVisualizationOptions.CreateDefault());

        var overlay = renderer.CreateOverlay(bitmap, Array.Empty<TableRegion>());

        Assert.Null(overlay);
    }

    [Fact]
    public void OverlayRenderer_DrawsRectanglesOnCopy()
    {
        using var bitmap = new SKBitmap(10, 10);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        var options = new TableVisualizationOptions(SKColors.Red, 1f);
        var renderer = new OverlayRenderer(options);
        var regions = new List<TableRegion> { new(0, 0, 5, 5, "cell") };

        using var overlay = renderer.CreateOverlay(bitmap, regions);

        Assert.NotNull(overlay);
        Assert.Equal(bitmap.Width, overlay.Width);
        Assert.Equal(bitmap.Height, overlay.Height);
        Assert.Equal(SKColors.White, bitmap.GetPixel(0, 0));
        Assert.Equal(options.StrokeColor, overlay.GetPixel(0, 0));
    }

    [Fact]
    public void TableStructureResult_ValidatesArgumentsAndStoresProperties()
    {
        using var overlay = new SKBitmap(1, 1);
        var regions = new List<TableRegion> { new(1, 2, 3, 4, "cell") };
        var snapshot = new TableFormerPerformanceSnapshot(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, 1, 1, 5, 5, 5);
        var result = new TableStructureResult(regions, overlay, TableFormerLanguage.French, TableFormerRuntime.OpenVino, TimeSpan.FromMilliseconds(10), snapshot);

        Assert.Same(regions, result.Regions);
        Assert.Same(overlay, result.OverlayImage);
        Assert.Equal(TableFormerLanguage.French, result.Language);
        Assert.Equal(TableFormerRuntime.OpenVino, result.Runtime);
        Assert.Equal(TimeSpan.FromMilliseconds(10), result.InferenceTime);
        Assert.Same(snapshot, result.PerformanceSnapshot);

        Assert.Throws<ArgumentNullException>(() => new TableStructureResult(null!, overlay, TableFormerLanguage.English, TableFormerRuntime.Onnx, TimeSpan.Zero, snapshot));
        Assert.Throws<ArgumentNullException>(() => new TableStructureResult(regions, overlay, TableFormerLanguage.English, TableFormerRuntime.Onnx, TimeSpan.Zero, null!));
    }
}
