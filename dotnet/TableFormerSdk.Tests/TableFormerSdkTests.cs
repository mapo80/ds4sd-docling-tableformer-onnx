using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TableFormerSdk;
using TableFormerSdk.Backends;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;
using TableFormerClient = TableFormerSdk.TableFormer;
using Xunit;

namespace TableFormerSdk.Tests;

file sealed class FakeBackend : ITableFormerBackend
{
    private readonly TimeSpan _delay;

    public FakeBackend(TimeSpan? delay = null)
    {
        _delay = delay ?? TimeSpan.Zero;
    }

    public int InvocationCount { get; private set; }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        InvocationCount++;
        if (_delay > TimeSpan.Zero)
        {
            Thread.Sleep(_delay);
        }

        return new List<TableRegion> { new(1, 1, 2, 2, "cell") };
    }
}

public class TableFormerSdkTests
{
    private static TableFormerClient CreateSdkWithFakeBackend(TableFormerPerformanceOptions? performanceOptions = null)
    {
        var paths = CreateTempVariantPaths();
        var options = new TableFormerSdkOptions(new TableFormerModelPaths(paths), performanceOptions: performanceOptions);

        var sdk = new TableFormerClient(options);
        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, new FakeBackend());
        return sdk;
    }

    private static TableFormerVariantModelPaths CreateTempVariantPaths()
    {
        static string CreateTempFile(string extension, string? contents = null)
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + extension);
            if (contents is null)
            {
                File.WriteAllBytes(path, Array.Empty<byte>());
            }
            else
            {
                File.WriteAllText(path, contents);
            }

            return path;
        }

        var encoder = CreateTempFile(".onnx");
        var tagEncoder = CreateTempFile(".onnx");
        var decoderStep = CreateTempFile(".onnx");
        var bboxDecoder = CreateTempFile(".onnx");
        // Minimal JSON payloads to satisfy validation when parsed during tests
        var config = CreateTempFile(".json", "{}");
        var wordMap = CreateTempFile(".json", "{}");

        return new TableFormerVariantModelPaths(
            encoder,
            tagEncoder,
            decoderStep,
            bboxDecoder,
            config,
            wordMap);
    }

    private static string CreateSampleImage()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".png");
        using var bitmap = new SKBitmap(16, 16);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);
        using var paint = new SKPaint { Color = SKColors.Black, StrokeWidth = 1f };
        canvas.DrawRect(new SKRect(2, 2, 14, 14), paint);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.Open(path, FileMode.Create, FileAccess.Write);
        data.SaveTo(stream);
        return path;
    }

    [Fact]
    public void Process_EmptyPath_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<ArgumentException>(() => sdk.Process("", false, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx));
    }

    [Fact]
    public void Process_MissingImage_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<FileNotFoundException>(() => sdk.Process("missing.png", false, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx));
    }

    [Fact]
    public void Process_Overlay_ReturnsBitmap()
    {
        var sdk = CreateSdkWithFakeBackend();
        var imagePath = CreateSampleImage();
        var result = sdk.Process(imagePath, true, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx);
        Assert.NotNull(result.OverlayImage);
        Assert.Single(result.Regions);
        Assert.Equal(TableFormerRuntime.Onnx, result.Runtime);
        Assert.True(result.InferenceTime >= TimeSpan.Zero);
        Assert.Equal(TableFormerRuntime.Onnx, result.PerformanceSnapshot.Runtime);
    }

    [Fact]
    public void Process_NoOverlay_ReturnsNull()
    {
        var sdk = CreateSdkWithFakeBackend();
        var imagePath = CreateSampleImage();
        var result = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx);
        Assert.Null(result.OverlayImage);
    }

    [Fact]
    public void Process_AutoRuntime_ExploresAndSelectsFastestBackend()
    {
        var performanceOptions = new TableFormerPerformanceOptions(minimumSamples: 1, slidingWindowSize: 8, runtimePriority: new[] { TableFormerRuntime.Onnx });
        var sdk = CreateSdkWithFakeBackend(performanceOptions);

        var onnxBackendSlow = new FakeBackend(TimeSpan.FromMilliseconds(15));
        var onnxBackendFast = new FakeBackend(TimeSpan.FromMilliseconds(2));

        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, onnxBackendSlow);
        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Accurate, onnxBackendFast);

        var imagePath = CreateSampleImage();

        var first = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);
        var second = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);
        var third = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);

        Assert.Equal(TableFormerRuntime.Onnx, first.Runtime);
        Assert.Equal(TableFormerRuntime.Onnx, second.Runtime);
        Assert.Equal(TableFormerRuntime.Onnx, third.Runtime);

        Assert.True(second.InferenceTime <= first.InferenceTime);
        Assert.True(third.InferenceTime <= second.InferenceTime);

        var snapshots = sdk.GetPerformanceSnapshots(TableFormerModelVariant.Fast);
        Assert.NotEmpty(snapshots);
        Assert.All(snapshots, s => Assert.Equal(TableFormerRuntime.Onnx, s.Runtime));
    }
}
