using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TableFormerSdk.Configuration;
using TableFormerSdk.Constants;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using Xunit;

namespace TableFormerSdk.Tests;

public class ConfigurationTests
{
    [Fact]
    public void TableVisualizationOptions_InvalidStrokeWidth_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TableVisualizationOptions(SKColors.Black, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TableVisualizationOptions(SKColors.Black, -1));
    }

    [Fact]
    public void TableVisualizationOptions_Default_UsesConstants()
    {
        var options = TableVisualizationOptions.CreateDefault();

        Assert.Equal(TableFormerConstants.DefaultOverlayColor, options.StrokeColor);
        Assert.Equal(TableFormerConstants.DefaultOverlayStrokeWidth, options.StrokeWidth);
    }

    [Fact]
    public void ReleaseModelCatalog_ReportsAvailabilityBasedOnArtifacts()
    {
        var directory = CreateTempDirectory();
        File.WriteAllBytes(Path.Combine(directory, "tableformer-fast-encoder.onnx"), Array.Empty<byte>());
        File.WriteAllBytes(Path.Combine(directory, "tableformer-accurate-encoder.onnx"), Array.Empty<byte>());

        var fastXmlPath = Path.Combine(directory, "tableformer-fast-encoder.xml");
        File.WriteAllBytes(fastXmlPath, Array.Empty<byte>());
        var fastBinPath = Path.ChangeExtension(fastXmlPath, ".bin");
        File.WriteAllBytes(fastBinPath!, Array.Empty<byte>());

        var accurateXmlPath = Path.Combine(directory, "tableformer-accurate-encoder.xml");
        File.WriteAllBytes(accurateXmlPath, Array.Empty<byte>());
        var accurateBinPath = Path.ChangeExtension(accurateXmlPath, ".bin");
        File.WriteAllBytes(accurateBinPath!, Array.Empty<byte>());

        var catalog = new ReleaseModelCatalog(directory);

        Assert.True(catalog.SupportsRuntime(TableFormerRuntime.Onnx));
        Assert.True(catalog.SupportsVariant(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast));
        Assert.True(catalog.SupportsVariant(TableFormerRuntime.Onnx, TableFormerModelVariant.Accurate));
        Assert.True(catalog.SupportsRuntime(TableFormerRuntime.OpenVino));
        Assert.True(catalog.SupportsVariant(TableFormerRuntime.OpenVino, TableFormerModelVariant.Accurate));

        var artifact = catalog.GetArtifact(TableFormerRuntime.OpenVino, TableFormerModelVariant.Accurate);
        Assert.Equal(accurateXmlPath, artifact.ModelPath);
        Assert.Equal(accurateBinPath, artifact.WeightsPath);
    }

    [Fact]
    public void ReleaseModelCatalog_MissingWeights_Throws()
    {
        var directory = CreateTempDirectory();
        File.WriteAllBytes(Path.Combine(directory, "tableformer-fast-encoder.onnx"), Array.Empty<byte>());
        var xmlPath = Path.Combine(directory, "tableformer-fast-encoder.xml");
        File.WriteAllBytes(xmlPath, Array.Empty<byte>());

        var catalog = new ReleaseModelCatalog(directory);

        Assert.True(catalog.SupportsRuntime(TableFormerRuntime.Onnx));
        Assert.False(catalog.SupportsVariant(TableFormerRuntime.OpenVino, TableFormerModelVariant.Fast));
        Assert.Throws<FileNotFoundException>(() => catalog.GetArtifact(TableFormerRuntime.OpenVino, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void TableFormerSdkOptions_BuildsLanguagesAndRuntimes()
    {
        var artifacts = new[]
        {
            new TableFormerModelArtifact(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, CreateTempFile(".onnx")),
            new TableFormerModelArtifact(TableFormerRuntime.Ort, TableFormerModelVariant.Fast, CreateTempFile(".ort"))
        };

        var options = new TableFormerSdkOptions(
            new FakeModelCatalog(artifacts),
            TableFormerLanguage.Italian,
            new[] { TableFormerLanguage.German, TableFormerLanguage.Italian });

        Assert.Equal(TableFormerLanguage.Italian, options.DefaultLanguage);
        Assert.Contains(TableFormerLanguage.Italian, options.SupportedLanguages);
        Assert.Contains(TableFormerLanguage.German, options.SupportedLanguages);
        Assert.Contains(TableFormerRuntime.Ort, options.AvailableRuntimes);
        Assert.Contains(TableFormerRuntime.Onnx, options.AvailableRuntimes);
        Assert.DoesNotContain(TableFormerRuntime.OpenVino, options.AvailableRuntimes);
    }

    [Fact]
    public void TableFormerSdkOptions_EnsureLanguageIsSupported_Throws()
    {
        var options = new TableFormerSdkOptions(new FakeModelCatalog(new[]
        {
            new TableFormerModelArtifact(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, CreateTempFile(".onnx"))
        }));

        Assert.Throws<NotSupportedException>(() => options.EnsureLanguageIsSupported(TableFormerLanguage.German));
    }

    private static string CreateTempFile(string extension)
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + extension);
        File.WriteAllBytes(path, Array.Empty<byte>());
        return path;
    }

    private static string CreateTempDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}

file sealed class FakeModelCatalog : ITableFormerModelCatalog
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
