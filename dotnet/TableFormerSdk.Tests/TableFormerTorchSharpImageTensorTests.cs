using System;
using System.IO;
using System.Threading.Tasks;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Tensorization;
using Xunit;

namespace TableFormerSdk.Tests;

public class TableFormerTorchSharpImageTensorTests
{
    [Fact]
    public async Task ImageTensorizationMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var referencePath = Path.Combine(repoRoot, "results", "tableformer_image_tensors_reference.json");
        Assert.True(File.Exists(referencePath), $"Reference file not found at '{referencePath}'.");

        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var reference = TableFormerImageTensorReference.Load(referencePath);

        using var bootstrapper = new TableFormerArtifactBootstrapper(
            new DirectoryInfo(Path.Combine(repoRoot, "dotnet", "artifacts_test_cache")));

        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
        var cropper = new TableFormerTableCropper();

        foreach (var sample in reference.Samples)
        {
            var imagePath = Path.Combine(datasetDir, sample.ImageName);
            Assert.True(File.Exists(imagePath), $"Image '{sample.ImageName}' not found in dataset.");

            var resizeSnapshot = cropper.PrepareTableCrops(new FileInfo(imagePath));
            Assert.InRange(sample.TableIndex, 0, resizeSnapshot.TableCrops.Count - 1);

            var cropSnapshot = resizeSnapshot.TableCrops[sample.TableIndex];

            using var tensorSnapshot = tensorizer.CreateTensor(cropSnapshot);

            Assert.Equal(reference.TargetSize, tensorSnapshot.TargetSize);
            Assert.Equal(reference.Channels, tensorSnapshot.Channels);
            Assert.Equal(1, tensorSnapshot.BatchSize);

            var tensorShape = tensorSnapshot.Tensor.shape;
            Assert.Equal(sample.TensorShape.Count, tensorShape.Length);
            for (var i = 0; i < tensorShape.Length; i++)
            {
                Assert.Equal(sample.TensorShape[i], tensorShape[i]);
            }

            Assert.Equal(sample.TensorSha256, tensorSnapshot.TensorSha256);
            Assert.Equal(sample.TensorMin, tensorSnapshot.MinValue, 6);
            Assert.Equal(sample.TensorMax, tensorSnapshot.MaxValue, 6);
            Assert.Equal(sample.TensorMean, tensorSnapshot.Mean, 6);
            Assert.Equal(sample.TensorStd, tensorSnapshot.StandardDeviation, 6);

            var expectedValues = sample.DecodeTensorValues();
            var actualValues = tensorSnapshot.GetTensorValuesCopy();

            Assert.Equal(expectedValues.Length, actualValues.Length);
            for (var i = 0; i < expectedValues.Length; i++)
            {
                Assert.Equal(expectedValues[i], actualValues[i], 6);
            }
        }
    }

    private static string GetRepositoryRoot()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var potentialRoot = Path.Combine(baseDirectory, "..", "..", "..", "..", "..");
        return Path.GetFullPath(potentialRoot);
    }
}
