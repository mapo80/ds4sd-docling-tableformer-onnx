using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Results;
using TableFormerTorchSharpSdk.Tensorization;
using TableFormerTorchSharpSdk.Utilities;
using Xunit;

namespace TableFormerSdk.Tests;

public sealed class TableFormerTorchSharpDoclingResponseTests
{
    [Fact]
    public async Task DoclingMultiOutputMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var referencePath = Path.Combine(repoRoot, "results", "tableformer_docling_fintabnet.json");
        Assert.True(File.Exists(referencePath), $"Reference JSON missing at '{referencePath}'.");
        var reference = TableFormerDoclingReference.Load(referencePath);

        using var bootstrapper = new TableFormerArtifactBootstrapper(
            new DirectoryInfo(Path.Combine(repoRoot, "dotnet", "artifacts_test_cache")));
        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync();

        using var neuralModel = new TableFormerTorchSharpSdk.Model.TableFormerNeuralModel(
            bootstrapResult.ConfigSnapshot,
            initializationSnapshot,
            bootstrapResult.ModelDirectory);

        var decoder = new TableFormerTorchSharpSdk.Decoding.TableFormerSequenceDecoder(initializationSnapshot);
        var cellMatcher = new TableFormerCellMatcher(bootstrapResult.ConfigSnapshot);
        var cropper = new TableFormerTableCropper();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
        var preparer = new TableFormerPageInputPreparer();
        var postProcessor = new TableFormerMatchingPostProcessor();
        var assembler = new TableFormerDoclingResponseAssembler();

        foreach (var page in reference.Pages.Values)
        {
            var imagePath = Path.Combine(datasetDir, page.ImageName);
            Assert.True(File.Exists(imagePath), $"Image '{page.ImageName}' not found in dataset.");

            var imageFile = new FileInfo(imagePath);
            var pageSnapshot = preparer.PreparePageInput(imageFile);
            var cropSnapshot = cropper.PrepareTableCrops(imageFile, pageSnapshot.TableBoundingBoxes);

            Assert.Equal(page.NumTables, cropSnapshot.TableCrops.Count);

            for (var tableIndex = 0; tableIndex < page.NumTables; tableIndex++)
            {
                var crop = cropSnapshot.TableCrops[tableIndex];
                using var tensorSnapshot = tensorizer.CreateTensor(crop);
                using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);

                var decoded = decoder.Decode(prediction);
                var matchingResult = cellMatcher.MatchCells(pageSnapshot, crop, decoded);
                var matchingDetails = matchingResult.ToMatchingDetails();
                var processed = pageSnapshot.Tokens.Count > 0
                    ? postProcessor.Process(matchingDetails.ToMutable(), correctOverlappingCells: false)
                    : matchingDetails;
                var assembled = assembler.Assemble(processed, decoded, sortRowColIndexes: true);

                var actualJson = ToCanonicalJson(assembled.ToDictionary());
                var expectedJson = reference.TablesByKey[(page.ImageName, tableIndex)];

                Assert.Equal(expectedJson, actualJson);
            }
        }
    }

    private static string ToCanonicalJson(IDictionary<string, object?> data)
    {
        var json = JsonSerializer.Serialize(data);
        using var document = JsonDocument.Parse(json);
        return JsonCanonicalizer.GetCanonicalJson(document.RootElement);
    }

    private static string GetRepositoryRoot()
    {
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
    }
}
