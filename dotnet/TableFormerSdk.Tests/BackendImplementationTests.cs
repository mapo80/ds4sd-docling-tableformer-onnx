using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using TableFormerSdk.Backends;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using Xunit;

namespace TableFormerSdk.Tests;

public class BackendImplementationTests
{
    [Fact]
    public void TableFormerOnnxBackend_NormalizesPixelsAndParsesRegions()
    {
        using var bitmap = new SKBitmap(2, 2);
        bitmap.SetPixel(0, 0, new SKColor(255, 128, 0));
        bitmap.SetPixel(1, 0, new SKColor(64, 64, 255));
        bitmap.SetPixel(0, 1, new SKColor(0, 0, 0));
        bitmap.SetPixel(1, 1, new SKColor(255, 255, 255));

        var adapter = new FakeOnnxSessionAdapter();
        var backend = new TableFormerOnnxBackend(adapter);

        var regions = backend.Infer(bitmap, "test");
        backend.Dispose();

        Assert.Single(regions);
        var region = regions[0];
        Assert.Equal("class_1", region.Label);
        Assert.Equal(0, region.X);
        Assert.True(region.Width > 0);
        Assert.InRange(adapter.LastTensor[0], 0.99f, 1f); // red channel normalized
        Assert.True(adapter.Disposed);
    }

    [Fact]
    public void OpenVinoBackend_WithBoxes_ParsesRegions()
    {
        using var bitmap = new SKBitmap(4, 4);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        var adapter = new FakeOpenVinoAdapter(withBoxes: true);
        using var backend = new OpenVinoBackend(adapter);

        var regions = backend.Infer(bitmap, "source");

        Assert.Single(regions);
        var region = regions[0];
        Assert.Equal("class_1", region.Label);
        Assert.True(region.Width > 0);
        Assert.True(region.Height > 0);
    }

    [Fact]
    public void OpenVinoBackend_WithoutBoxes_ReturnsFallbackRegion()
    {
        using var bitmap = new SKBitmap(4, 4);
        var adapter = new FakeOpenVinoAdapter(withBoxes: false);
        using var backend = new OpenVinoBackend(adapter);

        var regions = backend.Infer(bitmap, "source");

        Assert.Single(regions);
        var region = regions[0];
        Assert.Equal("openvino-region", region.Label);
        Assert.Equal(bitmap.Width, region.Width);
        Assert.Equal(bitmap.Height, region.Height);
    }

    [Fact]
    public void DefaultBackendFactory_InvalidRuntime_Throws()
    {
        var catalog = new FakeModelCatalog(new[]
        {
            new TableFormerModelArtifact(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, CreateTempModelFile(".onnx"))
        });
        var options = new TableFormerSdkOptions(catalog);
        var factory = new DefaultBackendFactory(options);

        Assert.Throws<ArgumentException>(() => factory.CreateBackend(TableFormerRuntime.Auto, TableFormerModelVariant.Fast));
        Assert.Throws<NotSupportedException>(() => factory.CreateBackend(TableFormerRuntime.Ort, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void DefaultBackendFactory_OpenVinoWithoutConfiguration_Throws()
    {
        var catalog = new FakeModelCatalog(new[]
        {
            new TableFormerModelArtifact(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, CreateTempModelFile(".onnx"))
        });
        var options = new TableFormerSdkOptions(catalog);
        var factory = new DefaultBackendFactory(options);

        Assert.Throws<NotSupportedException>(() => factory.CreateBackend(TableFormerRuntime.OpenVino, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void TableFormerOrtBackend_DelegatesToInnerBackend()
    {
        using var bitmap = new SKBitmap(2, 2);
        bitmap.SetPixel(0, 0, SKColors.White);
        bitmap.SetPixel(1, 0, SKColors.Black);

        var adapter = new FakeOnnxSessionAdapter();
        var inner = new TableFormerOnnxBackend(adapter);
        using (var backend = new TableFormerOrtBackend(inner))
        {
            var regions = backend.Infer(bitmap, "test");

            Assert.Single(regions);
        }

        Assert.True(adapter.Disposed);
    }

    private static string CreateTempModelFile(string extension)
    {
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), Guid.NewGuid().ToString("N") + extension);
        System.IO.File.WriteAllBytes(path, Array.Empty<byte>());
        return path;
    }

    private sealed class FakeOnnxSessionAdapter : IOnnxSessionAdapter
    {
        public float[] LastTensor { get; private set; } = Array.Empty<float>();
        public bool Disposed { get; private set; }

        public IEnumerable<OnnxOutput> Run(Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float> tensor)
        {
            LastTensor = tensor.ToArray();
            var logits = new[] { -5f, 5f };
            var boxes = new[] { 0.25f, 0.25f, 0.5f, 0.5f };
            yield return new OnnxOutput("logits", logits, new long[] { 1, 1, 2 });
            yield return new OnnxOutput("pred_boxes", boxes, new long[] { 1, 1, 4 });
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }

    private sealed class FakeOpenVinoAdapter : IOpenVinoRuntimeAdapter
    {
        private readonly bool _withBoxes;

        public FakeOpenVinoAdapter(bool withBoxes)
        {
            _withBoxes = withBoxes;
            ModelInputWidth = 4;
            ModelInputHeight = 4;
        }

        public int ModelInputWidth { get; }

        public int ModelInputHeight { get; }

        public bool Disposed { get; private set; }

        public OpenVinoOutputs Infer(float[] data, int height, int width)
        {
            var logits = new[] { -2f, 2f, 0f };
            var logitsShape = new long[] { 1, 1, 3 };
            if (!_withBoxes)
            {
                return new OpenVinoOutputs(logits, logitsShape, null, null);
            }

            var boxes = new[] { 0.2f, 0.3f, 0.4f, 0.5f };
            var boxesShape = new long[] { 1, 1, 4 };
            return new OpenVinoOutputs(logits, logitsShape, boxes, boxesShape);
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }

    private sealed class FakeModelCatalog : ITableFormerModelCatalog
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
}
