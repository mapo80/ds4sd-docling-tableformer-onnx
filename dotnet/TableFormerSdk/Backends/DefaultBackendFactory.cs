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
        TableFormerRuntime.Pipeline => _options.Pipeline is not null
            ? new TableFormerPipelineBackend(
                _options.Pipeline.ModelPaths.Encoder,
                _options.Pipeline.ModelPaths.BboxDecoder,
                _options.Pipeline.ModelPaths.Decoder)
            : throw new InvalidOperationException("Pipeline model paths are not configured"),
        TableFormerRuntime.OptimizedPipeline => _options.Pipeline is not null
            ? new TableFormerOptimizedPipelineBackend(
                _options.Pipeline.ModelPaths.Encoder,
                _options.Pipeline.ModelPaths.BboxDecoder)
            : throw new InvalidOperationException("Pipeline model paths are not configured"),
        _ => throw new ArgumentOutOfRangeException(nameof(runtime), runtime, "Unsupported runtime")
    };
}
