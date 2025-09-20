using System;
using System.IO;
using TableFormerSdk.Constants;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Configuration;

public sealed class TableFormerModelPaths
{
    public TableFormerModelPaths(string fastModelPath, string? accurateModelPath)
    {
        FastModelPath = ValidateModelPath(fastModelPath, nameof(fastModelPath));
        AccurateModelPath = accurateModelPath is null
            ? null
            : ValidateModelPath(accurateModelPath, nameof(accurateModelPath));
    }

    public string FastModelPath { get; }

    public string? AccurateModelPath { get; }

    public string GetModelPath(TableFormerModelVariant variant) => variant switch
    {
        TableFormerModelVariant.Fast => FastModelPath,
        TableFormerModelVariant.Accurate when AccurateModelPath is not null => AccurateModelPath,
        TableFormerModelVariant.Accurate => throw new InvalidOperationException(TableFormerConstants.AccurateModelNotConfiguredMessage),
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, TableFormerConstants.UnsupportedModelVariantMessage)
    };

    private static string ValidateModelPath(string path, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Model path is empty", argumentName);
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Model file not found: {path}", path);
        }

        return path;
    }
}
