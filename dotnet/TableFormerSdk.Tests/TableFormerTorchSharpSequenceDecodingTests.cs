using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Model;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Tensorization;
using Xunit;

namespace TableFormerSdk.Tests;

public class TableFormerTorchSharpSequenceDecodingTests
{
    [Fact]
    public async Task SequenceDecodingMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var tensorReferencePath = Path.Combine(repoRoot, "results", "tableformer_image_tensors_reference.json");
        Assert.True(File.Exists(tensorReferencePath), $"Tensor reference missing at '{tensorReferencePath}'.");
        var tensorReference = TableFormerImageTensorReference.Load(tensorReferencePath);

        var sequenceReferencePath = Path.Combine(repoRoot, "results", "tableformer_sequence_decoding_reference.json");
        Assert.True(File.Exists(sequenceReferencePath), $"Sequence decoding reference missing at '{sequenceReferencePath}'.");
        var sequenceReference = TableFormerSequenceDecodingReference.Load(sequenceReferencePath);

        using var bootstrapper = new TableFormerArtifactBootstrapper(
            new DirectoryInfo(Path.Combine(repoRoot, "dotnet", "artifacts_test_cache")));

        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync();

        using var neuralModel = new TableFormerNeuralModel(
            bootstrapResult.ConfigSnapshot,
            initializationSnapshot,
            bootstrapResult.ModelDirectory);

        var decoder = new TableFormerSequenceDecoder(initializationSnapshot);
        var cropper = new TableFormerTableCropper();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);

        var tensorSamples = tensorReference.Samples.ToDictionary(sample => (sample.ImageName, sample.TableIndex));

        foreach (var sample in sequenceReference.Samples)
        {
            Assert.True(
                tensorSamples.TryGetValue((sample.ImageName, sample.TableIndex), out var tensorSample),
                $"Tensor reference entry for {sample.ImageName}#{sample.TableIndex} was not found.");

            var imagePath = Path.Combine(datasetDir, sample.ImageName);
            Assert.True(File.Exists(imagePath), $"Image '{sample.ImageName}' not found in dataset.");

            var resizeSnapshot = cropper.PrepareTableCrops(new FileInfo(imagePath));
            Assert.InRange(sample.TableIndex, 0, resizeSnapshot.TableCrops.Count - 1);

            var cropSnapshot = resizeSnapshot.TableCrops[sample.TableIndex];
            using var tensorSnapshot = tensorizer.CreateTensor(cropSnapshot);

            Assert.Equal(tensorSample.TensorSha256, tensorSnapshot.TensorSha256);

            using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);
            var decoded = decoder.Decode(prediction);

            Assert.Equal(sample.TagSequence, decoded.TagSequence);
            Assert.Equal(sample.RsSequence, decoded.RsSequence);
            Assert.Equal(sample.HtmlSequence, decoded.HtmlSequence);
            Assert.Equal(sample.BoundingBoxesSynced, decoded.BoundingBoxesSynced);

            var rawValues = FlattenBoundingBoxes(decoded.RawBoundingBoxes);
            var finalValues = FlattenBoundingBoxes(decoded.FinalBoundingBoxes);

            AssertShape(sample.RawShape, rawValues);
            AssertShape(sample.FinalShape, finalValues);

            AssertFloatArraysAlmostEqual(sample.DecodeRawBoundingBoxes(), rawValues);
            AssertFloatArraysAlmostEqual(sample.DecodeFinalBoundingBoxes(), finalValues);

            VerifyStatistics(rawValues, sample.RawMin, sample.RawMax, sample.RawMean, sample.RawStd);
            VerifyStatistics(finalValues, sample.FinalMin, sample.FinalMax, sample.FinalMean, sample.FinalStd);
        }
    }

    private static void AssertShape(IReadOnlyList<int> shape, float[] values)
    {
        var expectedCount = shape.Aggregate(1, (product, dimension) => product * dimension);
        Assert.Equal(expectedCount, values.Length);
    }

    private static float[] FlattenBoundingBoxes(IReadOnlyList<TableFormerNormalizedBoundingBox> boxes)
    {
        var result = new float[boxes.Count * 4];
        for (var i = 0; i < boxes.Count; i++)
        {
            result[i * 4 + 0] = (float)boxes[i].Left;
            result[i * 4 + 1] = (float)boxes[i].Top;
            result[i * 4 + 2] = (float)boxes[i].Right;
            result[i * 4 + 3] = (float)boxes[i].Bottom;
        }

        return result;
    }

    private static void AssertFloatArraysAlmostEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        const float tolerance = 1.5e-7f;
        for (var i = 0; i < expected.Length; i++)
        {
            var delta = Math.Abs(expected[i] - actual[i]);
            Assert.True(delta <= tolerance, $"Index {i} delta {delta} exceeds tolerance {tolerance}.");
        }
    }

    private static void VerifyStatistics(
        float[] values,
        double? expectedMin,
        double? expectedMax,
        double? expectedMean,
        double? expectedStd)
    {
        if (values.Length == 0)
        {
            Assert.Null(expectedMin);
            Assert.Null(expectedMax);
            Assert.Null(expectedMean);
            Assert.Null(expectedStd);
            return;
        }

        var min = values.Min();
        var max = values.Max();
        var mean = values.Average();
        var variance = values.Select(v => Math.Pow(v - mean, 2)).Average();
        var std = Math.Sqrt(variance);

        if (expectedMin.HasValue)
        {
            Assert.Equal(expectedMin.Value, min, 5);
        }

        if (expectedMax.HasValue)
        {
            Assert.Equal(expectedMax.Value, max, 5);
        }

        if (expectedMean.HasValue)
        {
            Assert.Equal(expectedMean.Value, mean, 5);
        }

        if (expectedStd.HasValue)
        {
            Assert.Equal(expectedStd.Value, std, 5);
        }
    }

    private static string GetRepositoryRoot()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var potentialRoot = Path.Combine(baseDirectory, "..", "..", "..", "..", "..");
        return Path.GetFullPath(potentialRoot);
    }
}
