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
        TableFormerRuntime.Onnx => new TableFormerOnnxBackend(_options.Onnx.GetModelPath(variant)),
        TableFormerRuntime.Ort => throw new NotSupportedException("ORT backend not yet implemented"),
        TableFormerRuntime.OpenVino => _options.OpenVino is not null
            ? new OpenVinoBackend(_options.OpenVino.GetModelPaths(variant).Xml)
            : throw new InvalidOperationException("OpenVINO model paths are not configured"),
        _ => throw new ArgumentOutOfRangeException(nameof(runtime), runtime, "Unsupported runtime")
    };
}
