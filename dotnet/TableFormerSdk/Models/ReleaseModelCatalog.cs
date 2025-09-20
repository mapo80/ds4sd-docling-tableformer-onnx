using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Models;

public sealed class ReleaseModelCatalog : ITableFormerModelCatalog
{
    private static readonly IReadOnlyDictionary<TableFormerRuntime, IReadOnlyDictionary<TableFormerModelVariant, string>> ArtifactMap
        = new ReadOnlyDictionary<TableFormerRuntime, IReadOnlyDictionary<TableFormerModelVariant, string>>(new Dictionary<TableFormerRuntime, IReadOnlyDictionary<TableFormerModelVariant, string>>
        {
            [TableFormerRuntime.Onnx] = new ReadOnlyDictionary<TableFormerModelVariant, string>(new Dictionary<TableFormerModelVariant, string>
            {
                [TableFormerModelVariant.Fast] = "tableformer-fast-encoder.onnx",
                [TableFormerModelVariant.Accurate] = "tableformer-accurate-encoder.onnx"
            }),
            [TableFormerRuntime.OpenVino] = new ReadOnlyDictionary<TableFormerModelVariant, string>(new Dictionary<TableFormerModelVariant, string>
            {
                [TableFormerModelVariant.Fast] = "tableformer-fast-encoder.xml",
                [TableFormerModelVariant.Accurate] = "tableformer-accurate-encoder.xml"
            })
        });

    private readonly string _modelsDirectory;

    public static ReleaseModelCatalog CreateDefault()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var modelsDirectory = Path.Combine(baseDirectory, "models");
        return new ReleaseModelCatalog(modelsDirectory);
    }

    public ReleaseModelCatalog(string modelsDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelsDirectory))
        {
            throw new ArgumentException("Models directory is empty", nameof(modelsDirectory));
        }

        _modelsDirectory = modelsDirectory;
    }

    public bool SupportsRuntime(TableFormerRuntime runtime)
    {
        if (!ArtifactMap.ContainsKey(runtime))
        {
            return false;
        }

        return ArtifactMap[runtime].Values.Any(path => File.Exists(ResolvePath(path)) && EnsureWeights(runtime, ResolvePath(path), ensureExists: false));
    }

    public bool SupportsVariant(TableFormerRuntime runtime, TableFormerModelVariant variant)
    {
        if (!ArtifactMap.TryGetValue(runtime, out var variants))
        {
            return false;
        }

        if (!variants.TryGetValue(variant, out var relativePath))
        {
            return false;
        }

        var absolutePath = ResolvePath(relativePath);
        if (!File.Exists(absolutePath))
        {
            return false;
        }

        return EnsureWeights(runtime, absolutePath, ensureExists: false);
    }

    public TableFormerModelArtifact GetArtifact(TableFormerRuntime runtime, TableFormerModelVariant variant)
    {
        if (!ArtifactMap.TryGetValue(runtime, out var variants) || !variants.TryGetValue(variant, out var relativePath))
        {
            throw new NotSupportedException($"Runtime {runtime} variant {variant} is not available");
        }

        var absolutePath = ResolvePath(relativePath);
        if (!File.Exists(absolutePath))
        {
            throw new FileNotFoundException($"Model artifact not found for runtime {runtime} variant {variant}", absolutePath);
        }

        var weightsPath = EnsureWeights(runtime, absolutePath, ensureExists: true) ? ResolveWeightsPath(absolutePath) : null;
        return new TableFormerModelArtifact(runtime, variant, absolutePath, weightsPath);
    }

    private string ResolvePath(string relativePath)
        => Path.IsPathRooted(relativePath) ? relativePath : Path.Combine(_modelsDirectory, relativePath);

    private static bool EnsureWeights(TableFormerRuntime runtime, string modelPath, bool ensureExists)
    {
        if (runtime != TableFormerRuntime.OpenVino)
        {
            return true;
        }

        var weightsPath = ResolveWeightsPath(modelPath);
        if (!File.Exists(weightsPath))
        {
            if (ensureExists)
            {
                throw new FileNotFoundException($"Weights file not found for OpenVINO model {modelPath}", weightsPath);
            }

            return false;
        }

        return true;
    }

    private static string ResolveWeightsPath(string modelPath)
        => Path.ChangeExtension(modelPath, ".bin") ?? throw new InvalidOperationException($"Unable to compute weights path for {modelPath}");
}
