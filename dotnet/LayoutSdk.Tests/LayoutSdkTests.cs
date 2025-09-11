using LayoutSdk;
using System;
using System.IO;
using System.Collections.Generic;
using Xunit;
using SkiaSharp;

namespace LayoutSdk.Tests;

file class FakeBackend : ILayoutBackend
{
    public IReadOnlyList<BoundingBox> Infer(SKBitmap image) =>
        new List<BoundingBox> { new(1, 1, 2, 2, "box") };
}

public class LayoutSdkTests
{
    private static LayoutSdk CreateSdkWithFakeBackend()
    {
        var sdk = new LayoutSdk(new LayoutSdkOptions("", "", ""));
        sdk.Backends[LayoutEngine.Onnx] = new FakeBackend();
        return sdk;
    }

    [Fact]
    public void Process_EmptyPath_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<ArgumentException>(() => sdk.Process("", false, LayoutEngine.Onnx));
    }

    [Fact]
    public void Process_MissingImage_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<FileNotFoundException>(() => sdk.Process("missing.png", false, LayoutEngine.Onnx));
    }

    private static string SampleImage =>
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..","..","..","..","..","dataset", "gazette_de_france.jpg"));

    [Fact]
    public void Process_Overlay_ReturnsBitmap()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, true, LayoutEngine.Onnx);
        Assert.NotNull(result.OverlayImage);
        Assert.Single(result.Boxes);
    }

    [Fact]
    public void Process_NoOverlay_ReturnsNull()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, false, LayoutEngine.Onnx);
        Assert.Null(result.OverlayImage);
    }
}
