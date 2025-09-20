using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TableFormerSdk;
using TableFormerSdk.Backends;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;
using TableFormerClient = TableFormerSdk.TableFormerSdk;
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
    private static TableFormerClient CreateSdkWithFakeBackend(TableFormerPerformanceOptions? performanceOptions = null, bool includeOpenVino = false)
    {
        var artifacts = new List<TableFormerModelArtifact>
        {
            new(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, CreateTempModelFile(".onnx"))
        };

        if (includeOpenVino)
        {
            var xmlPath = CreateTempModelFile(".xml");
            var binPath = Path.ChangeExtension(xmlPath, ".bin");
            File.WriteAllBytes(binPath!, Array.Empty<byte>());
            artifacts.Add(new TableFormerModelArtifact(TableFormerRuntime.OpenVino, TableFormerModelVariant.Fast, xmlPath, binPath));
        }

        var options = new TableFormerSdkOptions(new FakeModelCatalog(artifacts), performanceOptions: performanceOptions);

        var sdk = new TableFormerClient(options);
        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, new FakeBackend());
        return sdk;
    }

    private static string CreateTempModelFile(string extension)
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + extension);
        File.WriteAllBytes(path, Array.Empty<byte>());
        return path;
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
    public void Process_WithOrtRuntime_UsesRegisteredBackend()
    {
        var catalog = new FakeModelCatalog(new[]
        {
            new TableFormerModelArtifact(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, CreateTempModelFile(".onnx")),
            new TableFormerModelArtifact(TableFormerRuntime.Ort, TableFormerModelVariant.Fast, CreateTempModelFile(".ort"))
        });
        var options = new TableFormerSdkOptions(catalog);
        using var sdk = new TableFormerClient(options);
        var backend = new FakeBackend();
        sdk.RegisterBackend(TableFormerRuntime.Ort, TableFormerModelVariant.Fast, backend);

        var imagePath = CreateSampleImage();
        var result = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Ort);

        Assert.Equal(1, backend.InvocationCount);
        Assert.Equal(TableFormerRuntime.Ort, result.Runtime);
    }

    [Fact]
    public void Process_AutoRuntime_ExploresAndSelectsFastestBackend()
    {
        var performanceOptions = new TableFormerPerformanceOptions(minimumSamples: 1, slidingWindowSize: 8, runtimePriority: new[] { TableFormerRuntime.Onnx, TableFormerRuntime.OpenVino });
        var sdk = CreateSdkWithFakeBackend(performanceOptions, includeOpenVino: true);

        var onnxBackend = new FakeBackend(TimeSpan.FromMilliseconds(15));
        var openVinoBackend = new FakeBackend(TimeSpan.FromMilliseconds(2));

        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, onnxBackend);
        sdk.RegisterBackend(TableFormerRuntime.OpenVino, TableFormerModelVariant.Fast, openVinoBackend);

        var imagePath = CreateSampleImage();

        var first = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);
        var second = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);
        var third = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);

        var observedRuntimes = new[] { first.Runtime, second.Runtime, third.Runtime };
        Assert.Contains(TableFormerRuntime.Onnx, observedRuntimes);
        Assert.Contains(TableFormerRuntime.OpenVino, observedRuntimes);
        Assert.Equal(TableFormerRuntime.OpenVino, third.Runtime);

        Assert.True(third.InferenceTime <= first.InferenceTime);

        var snapshots = sdk.GetPerformanceSnapshots(TableFormerModelVariant.Fast);
        Assert.Equal(2, snapshots.Count);
        Assert.Contains(snapshots, s => s.Runtime == TableFormerRuntime.Onnx && s.TotalSampleCount == 1);
        Assert.Contains(snapshots, s => s.Runtime == TableFormerRuntime.OpenVino && s.TotalSampleCount >= 2);
    }
}

file sealed class FakeModelCatalog : ITableFormerModelCatalog
{
    private readonly Dictionary<(TableFormerRuntime Runtime, TableFormerModelVariant Variant), TableFormerModelArtifact> _artifacts;

    public FakeModelCatalog(IEnumerable<TableFormerModelArtifact> artifacts)
    {
        _artifacts = artifacts.ToDictionary(a => (a.Runtime, a.Variant));
    }

    public bool SupportsRuntime(TableFormerRuntime runtime)
        => _artifacts.Keys.Any(key => key.Runtime == runtime);

    public bool SupportsVariant(TableFormerRuntime runtime, TableFormerModelVariant variant)
        => _artifacts.ContainsKey((runtime, variant));

    public TableFormerModelArtifact GetArtifact(TableFormerRuntime runtime, TableFormerModelVariant variant)
        => _artifacts.TryGetValue((runtime, variant), out var artifact)
            ? artifact
            : throw new NotSupportedException();
}
