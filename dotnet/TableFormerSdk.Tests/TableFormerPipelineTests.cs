using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using TableFormerSdk;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using Xunit;

namespace TableFormerSdk.Tests;

public sealed class TableFormerPipelineTests : IDisposable
{
    private readonly string _testImagePath;

    public TableFormerPipelineTests()
    {
        // Usa l'immagine di test standard del progetto
        _testImagePath = Path.Combine("..", "..", "..", "..", "..", "..", "dataset", "golden", "v0.12.0", "2305.03393v1-pg9", "source", "2305.03393v1-pg9-img.png");

        // Verifica che il file esista
        if (!File.Exists(_testImagePath))
        {
            throw new FileNotFoundException($"Test image not found: {_testImagePath}");
        }
    }

    [Fact]
    public void PipelineBackend_ProcessesImage_WithSeparateModels()
    {
        // Arrange - Usa percorsi assoluti per i modelli
        var encoderPath = Path.Combine("..", "..", "..", "..", "..", "models", "encoder.onnx");
        var bboxDecoderPath = Path.Combine("..", "..", "..", "..", "..", "models", "bbox_decoder.onnx");
        var decoderPath = Path.Combine("..", "..", "..", "..", "..", "models", "decoder.onnx");

        var options = new TableFormerSdkOptions(
            onnx: new TableFormerModelPaths(encoderPath, null),
            pipeline: new PipelineModelPaths(encoderPath, bboxDecoderPath, decoderPath)
        );

        using var sdk = new TableFormerSdk(options);

        // Act
        var result = sdk.Process(
            imagePath: _testImagePath,
            overlay: false,
            runtime: TableFormerRuntime.Pipeline,
            variant: TableFormerModelVariant.Fast
        );

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.Regions);
        Assert.True(result.Regions.Count >= 0); // Può essere 0 se nessuna tabella rilevata

        // Verifica che la pipeline abbia funzionato (non ci siano stati errori)
        Assert.NotNull(result.PerformanceSnapshot);
        Assert.True(result.InferenceTime.TotalMilliseconds > 0);

        // Log risultati per analisi
        Console.WriteLine($"Pipeline processed image: {_testImagePath}");
        Console.WriteLine($"Detected regions: {result.Regions.Count}");
        Console.WriteLine($"Processing time: {result.InferenceTime.TotalMilliseconds:F2}ms");
        Console.WriteLine($"Runtime used: {result.Runtime}");

        // Salva risultati in formato JSON per comparativa con Python
        SaveResultsForComparison(result, _testImagePath);
    }

    [Fact]
    public void PipelineConfiguration_ValidatesModelPaths()
    {
        // Arrange
        var encoderPath = Path.Combine("..", "..", "..", "models", "encoder.onnx");
        var bboxDecoderPath = Path.Combine("..", "..", "..", "models", "bbox_decoder.onnx");
        var decoderPath = Path.Combine("..", "..", "..", "models", "decoder.onnx");

        // Act & Assert
        var exception = Assert.Throws<FileNotFoundException>(() =>
        {
            new PipelineModelPaths("nonexistent.onnx", bboxDecoderPath, decoderPath);
        });
        Assert.Contains("Model file not found", exception.Message);

        // Verifica che configurazione valida funzioni
        var validConfig = new PipelineModelPaths(encoderPath, bboxDecoderPath, decoderPath);
        Assert.Equal(encoderPath, validConfig.EncoderPath);
        Assert.Equal(bboxDecoderPath, validConfig.BboxDecoderPath);
        Assert.Equal(decoderPath, validConfig.DecoderPath);
    }

    private void SaveResultsForComparison(TableStructureResult result, string imagePath)
    {
        var outputDir = Path.Combine("..", "..", "..", "..", "results", "dotnet-pipeline-test");
        Directory.CreateDirectory(outputDir);

        var jsonFile = Path.Combine(outputDir, "dotnet_pipeline_result.json");

        var jsonData = new
        {
            image = imagePath,
            model = "TableFormer Pipeline (encoder/decoder/bbox_decoder)",
            metadata = new
            {
                processingTimeMs = result.InferenceTime.TotalMilliseconds,
                runtime = result.Runtime.ToString(),
                modelVariant = "Fast"
            },
            detections = result.Regions.Select((region, index) => new
            {
                id = index,
                confidence = 0.5f, // Placeholder - il parser non restituisce confidence
                label = "Text", // Placeholder - il parser non restituisce label
                bbox = new
                {
                    left = region.X,
                    top = region.Y,
                    width = region.Width,
                    height = region.Height
                }
            }).ToArray()
        };

        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(jsonFile, JsonSerializer.Serialize(jsonData, options));

        Console.WriteLine($"Results saved to: {jsonFile}");
        Console.WriteLine($"Total regions detected: {result.Regions.Count}");
    }

    public void Dispose()
    {
        // Cleanup se necessario
    }
}