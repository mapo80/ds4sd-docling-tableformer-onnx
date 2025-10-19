using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Model;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Tensorization;
using TorchSharp;
using static TorchSharp.torch;
using TorchTensor = TorchSharp.torch.Tensor;
using Xunit;

namespace TableFormerSdk.Tests;

public class TableFormerTorchSharpNeuralInferenceTests
{
    [Fact]
    public async Task NeuralInferenceMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var tensorReferencePath = Path.Combine(repoRoot, "results", "tableformer_image_tensors_reference.json");
        Assert.True(File.Exists(tensorReferencePath), $"Tensor reference missing at '{tensorReferencePath}'.");
        var tensorReference = TableFormerImageTensorReference.Load(tensorReferencePath);

        var neuralReferencePath = Path.Combine(repoRoot, "results", "tableformer_neural_outputs_reference.json");
        Assert.True(File.Exists(neuralReferencePath), $"Neural reference missing at '{neuralReferencePath}'.");
        var neuralReference = TableFormerNeuralReference.Load(neuralReferencePath);

        using var bootstrapper = new TableFormerArtifactBootstrapper(
            new DirectoryInfo(Path.Combine(repoRoot, "dotnet", "artifacts_test_cache")));

        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync();

        using var neuralModel = new TableFormerNeuralModel(
            bootstrapResult.ConfigSnapshot,
            initializationSnapshot,
            bootstrapResult.ModelDirectory);

        var cropper = new TableFormerTableCropper();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);

        var tensorSamplesByKey = tensorReference.Samples
            .ToDictionary(sample => (sample.ImageName, sample.TableIndex));

        foreach (var neuralSample in neuralReference.Samples)
        {
            Assert.True(
                tensorSamplesByKey.TryGetValue((neuralSample.ImageName, neuralSample.TableIndex), out var tensorSample),
                $"Tensor reference entry for {neuralSample.ImageName}#{neuralSample.TableIndex} was not found.");

            var imagePath = Path.Combine(datasetDir, neuralSample.ImageName);
            Assert.True(File.Exists(imagePath), $"Image '{neuralSample.ImageName}' not found in dataset.");

            var resizeSnapshot = cropper.PrepareTableCrops(new FileInfo(imagePath));
            Assert.InRange(neuralSample.TableIndex, 0, resizeSnapshot.TableCrops.Count - 1);

            var cropSnapshot = resizeSnapshot.TableCrops[neuralSample.TableIndex];
            using var tensorSnapshot = tensorizer.CreateTensor(cropSnapshot);

            Assert.Equal(tensorSample.TensorSha256, tensorSnapshot.TensorSha256);

            using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);

            Assert.Equal(neuralSample.TagSequence.Count, prediction.Sequence.Count);
            for (var i = 0; i < neuralSample.TagSequence.Count; i++)
            {
                Assert.Equal(neuralSample.TagSequence[i], prediction.Sequence[i]);
            }

            Assert.Equal(neuralSample.ClassShape.Count, prediction.Classes.shape.Length);
            for (var i = 0; i < prediction.Classes.shape.Length; i++)
            {
                Assert.Equal(neuralSample.ClassShape[i], prediction.Classes.shape[i]);
            }

            Assert.Equal(neuralSample.CoordShape.Count, prediction.Coordinates.shape.Length);
            for (var i = 0; i < prediction.Coordinates.shape.Length; i++)
            {
                Assert.Equal(neuralSample.CoordShape[i], prediction.Coordinates.shape[i]);
            }

            var expectedClassValues = neuralSample.DecodeClassValues();
            var expectedCoordValues = neuralSample.DecodeCoordValues();

            Assert.Equal(neuralSample.ClassValueCount, expectedClassValues.Length);
            Assert.Equal(neuralSample.CoordValueCount, expectedCoordValues.Length);

            VerifyTensorParity(
                prediction.Classes,
                neuralSample.ClassShape,
                expectedClassValues,
                neuralSample.ClassSha256,
                neuralSample.ClassMin,
                neuralSample.ClassMax,
                neuralSample.ClassMean,
                neuralSample.ClassStd);

            VerifyTensorParity(
                prediction.Coordinates,
                neuralSample.CoordShape,
                expectedCoordValues,
                neuralSample.CoordSha256,
                neuralSample.CoordMin,
                neuralSample.CoordMax,
                neuralSample.CoordMean,
                neuralSample.CoordStd);
        }
    }

    private static void VerifyTensorParity(
        TorchTensor actual,
        IReadOnlyList<int> expectedShape,
        float[] expectedValues,
        string expectedSha256,
        double? expectedMin,
        double? expectedMax,
        double? expectedMean,
        double? expectedStd)
    {
        var shape = actual.shape;
        Assert.Equal(expectedShape.Count, shape.Length);
        for (var i = 0; i < shape.Length; i++)
        {
            Assert.Equal(expectedShape[i], shape[i]);
        }

        using var expectedTensor = CreateTensor(expectedValues, expectedShape);
        if (expectedValues.Length == 0)
        {
            Assert.Equal(0, actual.numel());
        }
        else
        {
            Assert.True(torch.allclose(actual, expectedTensor, rtol: 1e-5, atol: 1e-5));
            using var actualArgmax = actual.argmax(dim: actual.shape.Length - 1);
            using var expectedArgmax = expectedTensor.argmax(dim: expectedTensor.shape.Length - 1);
            using var actualArgmaxCpu = actualArgmax.to(torch.CPU);
            using var expectedArgmaxCpu = expectedArgmax.to(torch.CPU);
            var actualIndices = actualArgmaxCpu.data<long>().ToArray();
            var expectedIndices = expectedArgmaxCpu.data<long>().ToArray();
            Assert.Equal(expectedIndices.Length, actualIndices.Length);
            for (var i = 0; i < actualIndices.Length; i++)
            {
                Assert.Equal(expectedIndices[i], actualIndices[i]);
            }
        }

        using var actualFloat = actual.to_type(torch.ScalarType.Float32);
        using var actualCpu = actualFloat.to(torch.CPU);
        var actualValues = actualCpu.data<float>().ToArray();
        Assert.Equal(expectedValues.Length, actualValues.Length);

        var maxDelta = 0f;
        for (var i = 0; i < expectedValues.Length; i++)
        {
            var delta = Math.Abs(expectedValues[i] - actualValues[i]);
            if (delta > maxDelta)
            {
                maxDelta = delta;
            }
        }
        const float tolerance = 1e-5f;
        Assert.True(maxDelta <= tolerance, $"Maximum deviation {maxDelta} exceeds tolerance {tolerance}.");

        if (expectedValues.Length > 0)
        {
            var actualMin = actualValues.Min();
            var actualMax = actualValues.Max();
            var actualMean = actualValues.Average();
            var actualStd = Math.Sqrt(actualValues.Select(v => Math.Pow(v - actualMean, 2)).Average());

            Assert.Equal(expectedMin ?? actualMin, actualMin, 5);
            Assert.Equal(expectedMax ?? actualMax, actualMax, 5);
            Assert.Equal(expectedMean ?? actualMean, actualMean, 5);
            Assert.Equal(expectedStd ?? actualStd, actualStd, 5);
        }
        else
        {
            Assert.Null(expectedMin);
            Assert.Null(expectedMax);
            Assert.Null(expectedMean);
            Assert.Null(expectedStd);
        }
    }

    private static TorchTensor CreateTensor(float[] values, IReadOnlyList<int> shape)
    {
        if (values.Length == 0)
        {
            return torch.empty(shape.Select(i => (long)i).ToArray(), dtype: torch.ScalarType.Float32);
        }

        var dimensions = shape.Select(i => (long)i).ToArray();
        return torch.tensor(values, dimensions, dtype: torch.ScalarType.Float32);
    }

    private static string ComputeSha256(float[] values)
    {
        var bytes = MemoryMarshal.AsBytes(values.AsSpan());
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string GetRepositoryRoot()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var potentialRoot = Path.Combine(baseDirectory, "..", "..", "..", "..", "..");
        return Path.GetFullPath(potentialRoot);
    }
}
