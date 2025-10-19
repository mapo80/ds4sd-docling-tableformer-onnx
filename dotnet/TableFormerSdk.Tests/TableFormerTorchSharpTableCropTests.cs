using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using TableFormerTorchSharpSdk.PagePreparation;
using Xunit;

namespace TableFormerSdk.Tests;

public class TableFormerTorchSharpTableCropTests
{
    [Fact]
    public void TableCroppingMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var referencePath = Path.Combine(repoRoot, "results", "tableformer_table_crops_reference.json");
        Assert.True(File.Exists(referencePath), $"Reference file not found at '{referencePath}'.");

        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var reference = TableFormerTableCropReference.Load(referencePath);
        var cropper = new TableFormerTableCropper();

        foreach (var sample in reference.Samples)
        {
            var imagePath = Path.Combine(datasetDir, sample.ImageName);
            Assert.True(File.Exists(imagePath), $"Image '{sample.ImageName}' not found in dataset.");

            var snapshot = cropper.PrepareTableCrops(new FileInfo(imagePath));

            Assert.Equal((int)sample.OriginalWidth, snapshot.OriginalWidth);
            Assert.Equal((int)sample.OriginalHeight, snapshot.OriginalHeight);
            Assert.Equal(sample.ResizedWidth, snapshot.ResizedWidth);
            Assert.Equal(sample.ResizedHeight, snapshot.ResizedHeight);
            Assert.Equal(sample.ScaleFactor, snapshot.ScaleFactor, 10);

            Assert.Equal(sample.TableCrops.Count, snapshot.TableCrops.Count);

            for (var i = 0; i < sample.TableCrops.Count; i++)
            {
                var expected = sample.TableCrops[i];
                var actual = snapshot.TableCrops[i];

                Assert.Equal(expected.TableIndex, i);
                AssertBoundingBox(expected.OriginalBoundingBox, actual.OriginalBoundingBox.ToArray());
                AssertBoundingBox(expected.ScaledBoundingBox, actual.ScaledBoundingBox.ToArray());
                Assert.Equal(expected.RoundedBoundingBox.ToArray(), actual.RoundedBoundingBox.ToArray());

                Assert.Equal(expected.OriginalPixelWidth, actual.OriginalPixelWidth, 6);
                Assert.Equal(expected.OriginalPixelHeight, actual.OriginalPixelHeight, 6);
                Assert.Equal(expected.ScaledPixelWidth, actual.ScaledPixelWidth, 6);
                Assert.Equal(expected.ScaledPixelHeight, actual.ScaledPixelHeight, 6);
                Assert.Equal(expected.RoundedPixelWidth, actual.RoundedPixelWidth);
                Assert.Equal(expected.RoundedPixelHeight, actual.RoundedPixelHeight);

                Assert.Equal(expected.CropWidth, actual.RoundedPixelWidth);
                Assert.Equal(expected.CropHeight, actual.RoundedPixelHeight);

                Assert.Equal(expected.CropByteLength, actual.CropByteLength);
                Assert.Equal(expected.CropByteLength, actual.CropFloatLength);
                Assert.Equal(expected.CropSha256, actual.CropSha256);

                var (mean, std, channels) = ComputeStatistics(actual);
                Assert.Equal(expected.CropMean, mean, 6);
                Assert.Equal(expected.CropStd, std, 6);
                Assert.Equal(expected.CropChannels, channels);

                Assert.True(expected.CropByteLength > 0, "Crop byte length must be positive.");
                Assert.True(actual.CropBytes.SequenceEqual(actual.CropBytes), "Crop bytes should be self-consistent.");
            }
        }
    }

    private static void AssertBoundingBox(IReadOnlyList<double> expected, double[] actual)
    {
        Assert.Equal(expected.Count, actual.Length);
        for (var i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i], actual[i], 6);
        }
    }

    private static (double mean, double stdDev, int channels) ComputeStatistics(TableFormerTableCropSnapshot snapshot)
    {
        var bytes = snapshot.CropBytes;
        if (bytes.Length == 0)
        {
            return (0.0, 0.0, 0);
        }

        var pixelCount = snapshot.RoundedPixelWidth * snapshot.RoundedPixelHeight;
        if (pixelCount <= 0)
        {
            throw new InvalidDataException("Pixel count must be positive.");
        }

        var channels = bytes.Length / pixelCount;
        if (channels <= 0)
        {
            throw new InvalidDataException("Derived channel count must be positive.");
        }

        double sum = 0.0;
        foreach (var value in bytes)
        {
            sum += value;
        }

        var mean = sum / bytes.Length;

        double varianceSum = 0.0;
        foreach (var value in bytes)
        {
            var delta = value - mean;
            varianceSum += delta * delta;
        }

        var stdDev = Math.Sqrt(varianceSum / bytes.Length);
        return (mean, stdDev, channels);
    }

    private static string GetRepositoryRoot()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var potentialRoot = Path.Combine(baseDirectory, "..", "..", "..", "..", "..");
        return Path.GetFullPath(potentialRoot);
    }
}
