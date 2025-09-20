using System;
using System.IO;
using TableFormerSdk.Constants;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Configuration;

public sealed class OpenVinoModelPaths
{
    public OpenVinoModelPaths(string fastModelXmlPath, string? accurateModelXmlPath)
    {
        (FastModelXmlPath, FastModelWeightsPath) = ValidateModelPaths(fastModelXmlPath, nameof(fastModelXmlPath));
        if (accurateModelXmlPath is not null)
        {
            (AccurateModelXmlPath, AccurateModelWeightsPath) = ValidateModelPaths(accurateModelXmlPath, nameof(accurateModelXmlPath));
        }
    }

    public string FastModelXmlPath { get; }

    public string FastModelWeightsPath { get; }

    public string? AccurateModelXmlPath { get; }

    public string? AccurateModelWeightsPath { get; }

    public (string Xml, string Weights) GetModelPaths(TableFormerModelVariant variant) => variant switch
    {
        TableFormerModelVariant.Fast => (FastModelXmlPath, FastModelWeightsPath),
        TableFormerModelVariant.Accurate when AccurateModelXmlPath is not null && AccurateModelWeightsPath is not null
            => (AccurateModelXmlPath, AccurateModelWeightsPath),
        TableFormerModelVariant.Accurate => throw new InvalidOperationException(TableFormerConstants.AccurateModelNotConfiguredMessage),
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, TableFormerConstants.UnsupportedModelVariantMessage)
    };

    private static (string Xml, string Weights) ValidateModelPaths(string xmlPath, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(xmlPath))
        {
            throw new ArgumentException("Model path is empty", argumentName);
        }

        if (!File.Exists(xmlPath))
        {
            throw new FileNotFoundException($"Model file not found: {xmlPath}", xmlPath);
        }

        var weightsPath = Path.ChangeExtension(xmlPath, ".bin");
        if (!File.Exists(weightsPath))
        {
            throw new FileNotFoundException($"Weights file not found for OpenVINO model: {weightsPath}", weightsPath);
        }

        return (xmlPath, weightsPath);
    }
}
