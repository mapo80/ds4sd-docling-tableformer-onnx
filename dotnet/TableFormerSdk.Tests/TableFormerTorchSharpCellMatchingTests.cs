using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.Model;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Tensorization;
using Xunit;

namespace TableFormerSdk.Tests;

public class TableFormerTorchSharpCellMatchingTests
{

    [Fact]
    public async Task CellMatchingMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var cellReferencePath = Path.Combine(repoRoot, "results", "tableformer_cell_matching_reference.json");
        Assert.True(File.Exists(cellReferencePath), $"Cell matching reference missing at '{cellReferencePath}'.");
        var cellReference = TableFormerCellMatchingReference.Load(cellReferencePath);

        var pageReferencePath = Path.Combine(repoRoot, "results", "tableformer_page_input_reference.json");
        Assert.True(File.Exists(pageReferencePath), $"Page input reference missing at '{pageReferencePath}'.");
        var pageReference = TableFormerPageInputReference.Load(pageReferencePath);
        var pageSamples = pageReference.Samples.ToDictionary(sample => sample.ImageName);

        using var bootstrapper = new TableFormerArtifactBootstrapper(
            new DirectoryInfo(Path.Combine(repoRoot, "dotnet", "artifacts_test_cache")));

        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync();

        using var neuralModel = new TableFormerNeuralModel(
            bootstrapResult.ConfigSnapshot,
            initializationSnapshot,
            bootstrapResult.ModelDirectory);

        var decoder = new TableFormerSequenceDecoder(initializationSnapshot);
        var cellMatcher = new TableFormerCellMatcher(bootstrapResult.ConfigSnapshot);
        var cropper = new TableFormerTableCropper();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
        var preparer = new TableFormerPageInputPreparer();

        var pageSnapshots = new Dictionary<string, TableFormerPageInputSnapshot>(StringComparer.Ordinal);

        foreach (var sample in cellReference.Samples)
        {

            Assert.True(File.Exists(Path.Combine(datasetDir, sample.ImageName)),
                $"Image '{sample.ImageName}' not found in dataset.");

            if (!pageSnapshots.TryGetValue(sample.ImageName, out var pageSnapshot))
            {
                if (!pageSamples.TryGetValue(sample.ImageName, out var pageSample))
                {
                    throw new InvalidDataException($"Page reference missing sample for '{sample.ImageName}'.");
                }

                var boundingBoxes = pageSample.TableBoundingBoxes
                    .Select(b => new TableFormerBoundingBox(b[0], b[1], b[2], b[3]))
                    .ToArray();

                pageSnapshot = preparer.PreparePageInput(
                    new FileInfo(Path.Combine(datasetDir, sample.ImageName)),
                    boundingBoxes);
                pageSnapshots[sample.ImageName] = pageSnapshot;
            }

            var imageFile = new FileInfo(Path.Combine(datasetDir, sample.ImageName));
            var resizeSnapshot = cropper.PrepareTableCrops(imageFile, pageSnapshot.TableBoundingBoxes);
            Assert.InRange(sample.TableIndex, 0, resizeSnapshot.TableCrops.Count - 1);

            var cropSnapshot = resizeSnapshot.TableCrops[sample.TableIndex];
            using var tensorSnapshot = tensorizer.CreateTensor(cropSnapshot);

            using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);
            var decoded = decoder.Decode(prediction);
            var matchingResult = cellMatcher.MatchCells(pageSnapshot, cropSnapshot, decoded);

            Assert.Equal(sample.TableBoundingBox.Count, 4);
            AssertBoundingBox(sample.TableBoundingBox, matchingResult.TableBoundingBox);

            if (sample.PageWidth.HasValue)
            {
                Assert.Equal(sample.PageWidth.Value, matchingResult.PageWidth, 6);
            }

            if (sample.PageHeight.HasValue)
            {
                Assert.Equal(sample.PageHeight.Value, matchingResult.PageHeight, 6);
            }

            if (sample.IouThreshold.HasValue)
            {
                Assert.Equal(sample.IouThreshold.Value, matchingResult.IouThreshold, 6);
            }

            AssertPredictionBoundingBoxes(sample, matchingResult);
            AssertTableCells(sample.TableCells, matchingResult.TableCells);
            AssertMatches(sample.Matches, matchingResult.Matches);
            AssertPdfCells(sample.PdfCells, matchingResult.PdfCells);
        }
    }

    private static void AssertPredictionBoundingBoxes(
        TableFormerCellMatchingSample reference,
        TableFormerCellMatchingResult actual)
    {
        var expected = reference.PredictionBoundingBoxes;
        var actualFlattened = actual.PredictionBoundingBoxesPage
            .SelectMany(box => new[] { (float)box.Left, (float)box.Top, (float)box.Right, (float)box.Bottom })
            .ToArray();

        Assert.Equal(expected.Length, actualFlattened.Length);

        const float Tolerance = 5e-5f;
        var maxDelta = 0f;
        for (var i = 0; i < expected.Length; i++)
        {
            var delta = Math.Abs(expected[i] - actualFlattened[i]);
            if (delta > maxDelta)
            {
                maxDelta = delta;
            }

            Assert.True(delta <= Tolerance,
                $"Prediction bbox element {i} mismatch: expected {expected[i]}, actual {actualFlattened[i]}, |Δ|={delta} > {Tolerance}.");
        }

        Assert.True(maxDelta <= Tolerance,
            $"Maximum bbox delta {maxDelta} exceeds tolerance {Tolerance}.");
    }

    private static void AssertTableCells(
        IReadOnlyList<TableFormerCellReference> expected,
        IReadOnlyList<TableFormerTableCell> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var expectedCell = expected[i];
            var actualCell = actual[i];

            Assert.Equal(expectedCell.CellId, actualCell.CellId);
            Assert.Equal(expectedCell.RowId, actualCell.RowId);
            Assert.Equal(expectedCell.ColumnId, actualCell.ColumnId);
            AssertBoundingBox(expectedCell.BoundingBox, actualCell.BoundingBox);
            Assert.Equal(expectedCell.CellClass, actualCell.CellClass);
            Assert.Equal(expectedCell.Label, actualCell.Label);
            Assert.Equal(expectedCell.MulticolTag, actualCell.MulticolTag);
            Assert.Equal(expectedCell.Colspan, actualCell.Colspan);
            Assert.Equal(expectedCell.Rowspan, actualCell.Rowspan);
            Assert.Equal(expectedCell.ColspanValue, actualCell.ColspanValue);
            Assert.Equal(expectedCell.RowspanValue, actualCell.RowspanValue);
        }
    }

    private static void AssertMatches(
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> expected,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerCellMatch>> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        foreach (var pair in expected)
        {
            Assert.True(actual.TryGetValue(pair.Key, out var actualMatches),
                $"Missing matches for PDF cell '{pair.Key}'.");
            var expectedMatches = pair.Value;
            Assert.Equal(expectedMatches.Count, actualMatches.Count);
            for (var i = 0; i < expectedMatches.Count; i++)
            {
                Assert.Equal(expectedMatches[i].TableCellId, actualMatches[i].TableCellId);
                var expectedIopdf = expectedMatches[i].IntersectionOverPdf ?? 0.0;
                var actualIopdf = actualMatches[i].IntersectionOverPdf;
                Assert.True(Math.Abs(expectedIopdf - actualIopdf) <= 1e-6,
                    $"Match {pair.Key}[{i}] IOU differs: expected {expectedIopdf}, actual {actualIopdf}.");
            }
        }
    }

    private static void AssertPdfCells(
        IReadOnlyList<TableFormerPdfCellReference> expected,
        IReadOnlyList<TableFormerPdfCell> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            Assert.Equal(expected[i].Text, actual[i].Text);
            AssertBoundingBox(expected[i].BoundingBox, actual[i].BoundingBox);
        }
    }

    private static void AssertBoundingBox(IReadOnlyList<double> expected, TableFormerBoundingBox actual)
    {
        Assert.Equal(expected.Count, 4);

        const double BboxTolerance = 3e-5;
        Assert.InRange(Math.Abs(expected[0] - actual.Left), 0, BboxTolerance);
        Assert.InRange(Math.Abs(expected[1] - actual.Top), 0, BboxTolerance);
        Assert.InRange(Math.Abs(expected[2] - actual.Right), 0, BboxTolerance);
        Assert.InRange(Math.Abs(expected[3] - actual.Bottom), 0, BboxTolerance);
    }

    private static string GetRepositoryRoot()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var potentialRoot = Path.Combine(baseDirectory, "..", "..", "..", "..", "..");
        return Path.GetFullPath(potentialRoot);
    }
}
