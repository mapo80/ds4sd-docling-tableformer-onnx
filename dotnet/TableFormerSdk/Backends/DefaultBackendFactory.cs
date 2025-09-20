using System;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Backends;

internal sealed class DefaultBackendFactory : ITableFormerBackendFactory
{
    private readonly TableFormerSdkOptions _options;

    public DefaultBackendFactory(TableFormerSdkOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public ITableFormerBackend CreateBackend(TableFormerRuntime runtime, TableFormerModelVariant variant) => runtime switch
    {
        TableFormerRuntime.Auto => throw new ArgumentException("Auto runtime must be resolved before backend creation", nameof(runtime)),
        TableFormerRuntime.Onnx => CreateOnnxBackend(TableFormerRuntime.Onnx, variant),
        TableFormerRuntime.Ort => CreateOrtBackend(variant),
        TableFormerRuntime.OpenVino => CreateOpenVinoBackend(variant),
        _ => throw new ArgumentOutOfRangeException(nameof(runtime), runtime, "Unsupported runtime")
    };

    private ITableFormerBackend CreateOnnxBackend(TableFormerRuntime runtime, TableFormerModelVariant variant)
    {
        var artifact = _options.ModelCatalog.GetArtifact(runtime, variant);
        return new TableFormerOnnxBackend(artifact.ModelPath);
    }

    private ITableFormerBackend CreateOrtBackend(TableFormerModelVariant variant)
    {
        var artifact = _options.ModelCatalog.GetArtifact(TableFormerRuntime.Ort, variant);
        return new TableFormerOrtBackend(artifact.ModelPath);
    }

    private ITableFormerBackend CreateOpenVinoBackend(TableFormerModelVariant variant)
    {
        var artifact = _options.ModelCatalog.GetArtifact(TableFormerRuntime.OpenVino, variant);
        return new OpenVinoBackend(artifact.ModelPath);
    }
}
