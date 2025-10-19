using System.IO;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Initialization;
using Xunit;

namespace TableFormerSdk.Tests;

public class TableFormerTorchSharpInitializationTests
{
    [Fact]
    public async Task PredictorInitializationMatchesPythonReference()
    {
        var repoRoot = GetRepositoryRoot();
        var referencePath = Path.Combine(repoRoot, "results", "tableformer_init_fast_reference.json");
        Assert.True(File.Exists(referencePath), $"Reference file not found at '{referencePath}'.");

        using var bootstrapper = new TableFormerArtifactBootstrapper(
            new DirectoryInfo(Path.Combine(repoRoot, "dotnet", "artifacts_test_cache")));

        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync();

        var reference = await TableFormerInitializationReference.LoadAsync(referencePath);
        initializationSnapshot.EnsureMatches(reference);
    }

    private static string GetRepositoryRoot()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var potentialRoot = Path.Combine(baseDirectory, "..", "..", "..", "..", "..");
        return Path.GetFullPath(potentialRoot);
    }
}
