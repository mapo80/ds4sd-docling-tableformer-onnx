using System;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Models;

public sealed class TableFormerModelArtifact
{
    public TableFormerModelArtifact(TableFormerRuntime runtime, TableFormerModelVariant variant, string modelPath, string? weightsPath = null)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path is empty", nameof(modelPath));
        }

        Runtime = runtime;
        Variant = variant;
        ModelPath = modelPath;
        WeightsPath = weightsPath;
    }

    public TableFormerRuntime Runtime { get; }

    public TableFormerModelVariant Variant { get; }

    public string ModelPath { get; }

    public string? WeightsPath { get; }
}
