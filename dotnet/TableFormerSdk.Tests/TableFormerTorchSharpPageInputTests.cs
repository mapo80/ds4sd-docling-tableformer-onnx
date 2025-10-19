using System.IO;

using TableFormerTorchSharpSdk.PagePreparation;
using Xunit;

namespace TableFormerSdk.Tests;

public class TableFormerTorchSharpPageInputTests
{
    [Fact]
    public void PageInputMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var referencePath = Path.Combine(repoRoot, "results", "tableformer_page_input_reference.json");
        Assert.True(File.Exists(referencePath), $"Reference file not found at '{referencePath}'.");

        var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
        Assert.True(Directory.Exists(datasetDir), $"Dataset directory not found at '{datasetDir}'.");

        var reference = TableFormerPageInputReference.Load(referencePath);
        var preparer = new TableFormerPageInputPreparer();

        foreach (var sample in reference.Samples)
        {
            var imagePath = Path.Combine(datasetDir, sample.ImageName);
            Assert.True(File.Exists(imagePath), $"Image '{sample.ImageName}' not found in dataset.");

            var snapshot = preparer.PreparePageInput(new FileInfo(imagePath));

            Assert.Equal(sample.Width, snapshot.Width);
            Assert.Equal(sample.Height, snapshot.Height);
            Assert.Equal(sample.TokenCount, snapshot.Tokens.Count);
            Assert.Equal(sample.TableBoundingBoxes.Count, snapshot.TableBoundingBoxes.Count);

            for (var i = 0; i < sample.TableBoundingBoxes.Count; i++)
            {
                var expected = sample.TableBoundingBoxes[i];
                var actual = snapshot.TableBoundingBoxes[i].ToArray();
                Assert.Equal(expected.Length, actual.Length);
                for (var j = 0; j < expected.Length; j++)
                {
                    Assert.Equal(expected[j], actual[j], 6);
                }
            }

            Assert.Equal(sample.ImageSha256, snapshot.ImageSha256);
            Assert.Equal(sample.ImageBytes.Length, snapshot.ImageBytes.Length);
            Assert.True(sample.ImageBytes.AsSpan().SequenceEqual(snapshot.ImageBytes));
        }
    }

    private static string GetRepositoryRoot()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var potentialRoot = Path.Combine(baseDirectory, "..", "..", "..", "..", "..");
        return Path.GetFullPath(potentialRoot);
    }
}
