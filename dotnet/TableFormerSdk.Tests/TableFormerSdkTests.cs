using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using TableFormerSdk;
using TableFormerClient = TableFormerSdk.TableFormerSdk;
using Xunit;

namespace TableFormerSdk.Tests;

file sealed class FakeBackend : ITableFormerBackend
{
    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath) =>
        new List<TableRegion> { new(1, 1, 2, 2, "cell") };
}

public class TableFormerSdkTests
{
    private static TableFormerClient CreateSdkWithFakeBackend()
    {
        var sdk = new TableFormerClient(new TableFormerSdkOptions(new TableFormerModelPaths("", null)));
        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, new FakeBackend());
        return sdk;
    }

    [Fact]
    public void Process_EmptyPath_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<ArgumentException>(() => sdk.Process("", false, TableFormerRuntime.Onnx, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void Process_MissingImage_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<FileNotFoundException>(() => sdk.Process("missing.png", false, TableFormerRuntime.Onnx, TableFormerModelVariant.Fast));
    }

    private static string SampleImage
    {
        get
        {
            var datasetDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "dataset", "FinTabNet", "images"));
            if (!Directory.Exists(datasetDir))
            {
                throw new DirectoryNotFoundException($"Missing FinTabNet images directory: {datasetDir}. Run 'python scripts/extract_fintabnet_images.py dataset/FinTabNet/test-data-sample.parquet --output-dir dataset/FinTabNet/images' to generate it.");
            }

            var images = Directory.GetFiles(datasetDir, "*.png");
            if (images.Length == 0)
            {
                throw new InvalidOperationException("FinTabNet dataset extracted images not found. Re-run the extraction script to regenerate them.");
            }

            Array.Sort(images, StringComparer.OrdinalIgnoreCase);
            return images[0];
        }
    }

    [Fact]
    public void Process_Overlay_ReturnsBitmap()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, true, TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);
        Assert.NotNull(result.OverlayImage);
        Assert.Single(result.Regions);
    }

    [Fact]
    public void Process_NoOverlay_ReturnsNull()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, false, TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);
        Assert.Null(result.OverlayImage);
    }
}
