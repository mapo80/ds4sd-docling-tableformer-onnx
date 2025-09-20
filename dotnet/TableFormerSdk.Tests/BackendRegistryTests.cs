using SkiaSharp;
using System;
using System.Collections.Generic;
using TableFormerSdk.Backends;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using Xunit;

namespace TableFormerSdk.Tests;

public class BackendRegistryTests
{
    [Fact]
    public void GetOrCreateBackend_UsesFactoryAndCachesInstances()
    {
        var factory = new FakeFactory();
        using var registry = new BackendRegistry(factory);

        var first = registry.GetOrCreateBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);
        var second = registry.GetOrCreateBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);

        Assert.Same(first, second);
        Assert.Equal(1, factory.CreationRequests);
    }

    [Fact]
    public void RegisterBackend_WithAutoRuntime_Throws()
    {
        var factory = new FakeFactory();
        using var registry = new BackendRegistry(factory);

        Assert.Throws<ArgumentException>(() => registry.RegisterBackend(TableFormerRuntime.Auto, TableFormerModelVariant.Fast, new FakeBackend()));
    }

    [Fact]
    public void RegisterBackend_OverridesFactoryInstance()
    {
        var factory = new FakeFactory();
        using var registry = new BackendRegistry(factory);

        var backend = new FakeBackend();
        registry.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, backend);

        var resolved = registry.GetOrCreateBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);
        Assert.Same(backend, resolved);
    }

    [Fact]
    public void Dispose_InvokesBackendDispose()
    {
        var factory = new FakeFactory();
        var registry = new BackendRegistry(factory);

        var backend = (FakeBackend)registry.GetOrCreateBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);
        registry.Dispose();

        Assert.True(backend.Disposed);
    }

    [Fact]
    public void BackendKey_ToString_ContainsRuntimeAndVariant()
    {
        var key = new BackendKey(TableFormerRuntime.OpenVino, TableFormerModelVariant.Accurate);
        var text = key.ToString();

        Assert.Contains("OpenVino", text);
        Assert.Contains("Accurate", text);
    }

    private sealed class FakeFactory : ITableFormerBackendFactory
    {
        public int CreationRequests { get; private set; }

        public ITableFormerBackend CreateBackend(TableFormerRuntime runtime, TableFormerModelVariant variant)
        {
            CreationRequests++;
            return new FakeBackend();
        }
    }

    private sealed class FakeBackend : ITableFormerBackend, IDisposable
    {
        public bool Disposed { get; private set; }

        public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath) => Array.Empty<TableRegion>();

        public void Dispose()
        {
            Disposed = true;
        }
    }
}
