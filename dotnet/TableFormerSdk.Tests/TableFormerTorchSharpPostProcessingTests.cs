using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Matching;
using Xunit;

namespace TableFormerSdk.Tests;

public sealed class TableFormerTorchSharpPostProcessingTests
{
    [Fact]
    public async Task PostProcessingMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var postProcessingPath = Path.Combine(repoRoot, "results", "tableformer_post_processing_reference.json");
        Assert.True(File.Exists(postProcessingPath), $"Post-processing reference missing at '{postProcessingPath}'.");
        var postProcessingReference = TableFormerPostProcessingReference.Load(postProcessingPath);

        var cellReferencePath = Path.Combine(repoRoot, "results", "tableformer_cell_matching_reference.json");
        var cellReference = TableFormerCellMatchingReference.Load(cellReferencePath);

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
        var cropper = new TableFormerTorchSharpSdk.PagePreparation.TableFormerTableCropper();
        var tensorizer = TableFormerTorchSharpSdk.Tensorization.TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
        var preparer = new TableFormerTorchSharpSdk.PagePreparation.TableFormerPageInputPreparer();
        var processor = new TableFormerMatchingPostProcessor();

        foreach (var sample in postProcessingReference.Samples)
        {
            var cellSample = cellReference.SamplesByKey[(sample.ImageName, sample.TableIndex)];
            var imagePath = Path.Combine(datasetDir, sample.ImageName);
            Assert.True(File.Exists(imagePath), $"Image '{sample.ImageName}' not found in dataset.");

            var pageBoundingBoxes = cellSample.TableBoundingBox
                .Chunk(4)
                .Select(chunk => new TableFormerTorchSharpSdk.PagePreparation.TableFormerBoundingBox(chunk[0], chunk[1], chunk[2], chunk[3]))
                .ToArray();

            var pageSnapshot = preparer.PreparePageInput(new FileInfo(imagePath), pageBoundingBoxes);
            var cropSnapshot = cropper
                .PrepareTableCrops(new FileInfo(imagePath), pageSnapshot.TableBoundingBoxes)
                .TableCrops[sample.TableIndex];

            using var tensorSnapshot = tensorizer.CreateTensor(cropSnapshot);
            using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);
            var decoded = decoder.Decode(prediction);
            var matchingResult = cellMatcher.MatchCells(pageSnapshot, cropSnapshot, decoded);

            var mutableDetails = matchingResult.ToMatchingDetails().ToMutable();
            ReplaceTableCells(mutableDetails, sample.Input.TableCells);
            ReplacePdfCells(mutableDetails, sample.Input.PdfCells);
            ReplaceMatches(mutableDetails, sample.Input.Matches);

            if (sample.Input.MaxCellId >= 0)
            {
                Assert.Equal(sample.Input.MaxCellId, mutableDetails.TableCells.Max(cell => cell.CellId));
            }

            var processed = processor.Process(mutableDetails, correctOverlappingCells: false);

            AssertTableCells(sample.Output.TableCells, processed.TableCells);
            AssertMatches(sample.Output.Matches, processed.Matches);
            AssertPdfCells(sample.Output.PdfCells, processed.PdfCells);

            if (sample.Output.MaxCellId >= 0)
            {
                Assert.Equal(sample.Output.MaxCellId, processed.TableCells.Max(cell => cell.CellId));
            }

            var docOutput = GenerateDocOutput(processed.TableCells, processed.Matches);
            AssertDocOutput(sample.DocOutput, docOutput);
        }
    }

    private static void ReplaceTableCells(
        MutableTableFormerMatchingDetails mutable,
        IReadOnlyList<TableFormerCellReference> tableCells)
    {
        mutable.TableCells.Clear();
        foreach (var cell in tableCells)
        {
            var mutableCell = new TableFormerMutableTableCell(
                cell.CellId,
                cell.RowId,
                cell.ColumnId,
                cell.BoundingBox.ToArray(),
                cell.CellClass,
                cell.Label,
                cell.MulticolTag ?? string.Empty,
                cell.Colspan,
                cell.Rowspan);
            mutableCell.ColspanValue = cell.ColspanValue;
            mutableCell.RowspanValue = cell.RowspanValue;
            mutable.TableCells.Add(mutableCell);
        }
    }

    private static void ReplacePdfCells(
        MutableTableFormerMatchingDetails mutable,
        IReadOnlyList<TableFormerPdfCellReference> pdfCells)
    {
        mutable.PdfCells.Clear();
        foreach (var pdfCell in pdfCells)
        {
            mutable.PdfCells.Add(new TableFormerMutablePdfCell(
                pdfCell.Id,
                new TableFormerTorchSharpSdk.PagePreparation.TableFormerBoundingBox(
                    pdfCell.BoundingBox[0],
                    pdfCell.BoundingBox[1],
                    pdfCell.BoundingBox[2],
                    pdfCell.BoundingBox[3]),
                pdfCell.Text));
        }
    }

    private static void ReplaceMatches(
        MutableTableFormerMatchingDetails mutable,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> matches)
    {
        mutable.Matches.Clear();
        foreach (var pair in matches)
        {
            var list = new List<TableFormerMutableMatch>();
            foreach (var match in pair.Value)
            {
                list.Add(new TableFormerMutableMatch(
                    match.TableCellId,
                    match.IntersectionOverPdf,
                    match.IntersectionOverUnion,
                    match.PostScore));
            }

            mutable.Matches[pair.Key] = list;
        }
    }

    private static void AssertTableCells(
        IReadOnlyList<TableFormerCellReference> expected,
        IReadOnlyList<TableFormerMutableTableCell> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var exp = expected[i];
            var act = actual[i];
            Assert.Equal(exp.CellId, act.CellId);
            Assert.Equal(exp.RowId, act.RowId);
            Assert.Equal(exp.ColumnId, act.ColumnId);
            AssertBoundingBox(exp.BoundingBox, act.BoundingBox);
            Assert.Equal(exp.CellClass, act.CellClass);
            Assert.Equal(exp.Label, act.Label);
            Assert.Equal(exp.MulticolTag ?? string.Empty, act.MulticolTag);
            Assert.Equal(exp.Colspan, act.Colspan);
            Assert.Equal(exp.Rowspan, act.Rowspan);
            Assert.Equal(exp.ColspanValue, act.ColspanValue);
            Assert.Equal(exp.RowspanValue, act.RowspanValue);
        }
    }

    private static void AssertPdfCells(
        IReadOnlyList<TableFormerPdfCellReference> expected,
        IReadOnlyList<TableFormerMutablePdfCell> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var exp = expected[i];
            var act = actual[i];
            Assert.Equal(exp.Id, act.Id);
            Assert.Equal(exp.Text, act.Text);
            AssertBoundingBox(exp.BoundingBox, act.BoundingBox.ToArray());
        }
    }

    private static void AssertMatches(
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> expected,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        foreach (var pair in expected)
        {
            Assert.True(actual.TryGetValue(pair.Key, out var actualList), $"Missing matches for PDF cell '{pair.Key}'.");
            var expectedList = pair.Value;
            Assert.Equal(expectedList.Count, actualList.Count);
            for (var i = 0; i < expectedList.Count; i++)
            {
                var exp = expectedList[i];
                var act = actualList[i];
                Assert.Equal(exp.TableCellId, act.TableCellId);
                Assert.Equal(exp.IntersectionOverPdf ?? 0.0, act.IntersectionOverPdf ?? 0.0, 6);
                Assert.Equal(exp.IntersectionOverUnion ?? 0.0, act.IntersectionOverUnion ?? 0.0, 6);
                Assert.Equal(exp.PostScore ?? 0.0, act.PostScore ?? 0.0, 6);
            }
        }
    }

    private static void AssertDocOutput(
        IReadOnlyList<TableFormerDocOutputReference> expected,
        IReadOnlyList<TableFormerDocOutputReference> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var exp = expected[i];
            var act = actual[i];
            Assert.Equal(exp.PdfCellId, act.PdfCellId);
            Assert.Equal(exp.TableCellId, act.TableCellId);
            AssertBoundingBox(exp.BoundingBox, act.BoundingBox);
            Assert.Equal(exp.ColumnSpan, act.ColumnSpan);
            Assert.Equal(exp.RowSpan, act.RowSpan);
            Assert.Equal(exp.ColumnHeader, act.ColumnHeader);
            Assert.Equal(exp.RowHeader, act.RowHeader);
            Assert.Equal(exp.RowSection, act.RowSection);
        }
    }

    private static void AssertBoundingBox(IReadOnlyList<double> expected, IReadOnlyList<double> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i], actual[i], 6);
        }
    }

    private static IReadOnlyList<TableFormerDocOutputReference> GenerateDocOutput(
        IReadOnlyList<TableFormerMutableTableCell> tableCells,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>> matches)
    {
        var lookup = tableCells.ToDictionary(cell => cell.CellId, cell => cell);
        var output = new List<TableFormerDocOutputReference>();
        foreach (var pair in matches.OrderBy(pair => int.Parse(pair.Key, System.Globalization.CultureInfo.InvariantCulture)))
        {
            if (pair.Value.Count == 0)
            {
                continue;
            }

            var tableCellId = pair.Value[0].TableCellId;
            if (!lookup.TryGetValue(tableCellId, out var cell))
            {
                continue;
            }

            var bbox = cell.BoundingBox.ToArray();
            var label = cell.Label ?? string.Empty;
            var colSpan = cell.ColspanValue ?? cell.Colspan ?? 1;
            var rowSpan = cell.RowspanValue ?? cell.Rowspan ?? 1;

            output.Add(new TableFormerDocOutputReference(
                pair.Key,
                tableCellId,
                bbox,
                colSpan,
                rowSpan,
                label == "ched",
                label == "rhed",
                label == "srow"));
        }

        return output;
    }

    private static string GetRepositoryRoot()
    {
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
    }
}
