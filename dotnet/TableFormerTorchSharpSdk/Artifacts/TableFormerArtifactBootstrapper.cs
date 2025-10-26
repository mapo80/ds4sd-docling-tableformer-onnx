using System.IO;

using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Initialization;
using TableFormerTorchSharpSdk.Assets;

namespace TableFormerTorchSharpSdk.Artifacts;

public sealed class TableFormerArtifactBootstrapper : IDisposable
{
    public TableFormerArtifactBootstrapper(
        DirectoryInfo artifactsRoot,
        TableFormerModelVariant variant = TableFormerModelVariant.Fast,
        TableFormerGithubReleaseOptions? releaseOptions = null)
    {
        ArgumentNullException.ThrowIfNull(artifactsRoot);

        ArtifactsRoot = artifactsRoot;
        Variant = variant;
        ReleaseOptions = releaseOptions ?? new TableFormerGithubReleaseOptions();
    }

    public DirectoryInfo ArtifactsRoot { get; }

    public TableFormerModelVariant Variant { get; }

    public TableFormerGithubReleaseOptions ReleaseOptions { get; }

    public async Task<TableFormerBootstrapResult> EnsureArtifactsAsync(CancellationToken cancellationToken = default)
    {
        var downloadResult = await TableFormerReleaseDownloader.EnsureVariantAsync(
            Variant,
            ArtifactsRoot,
            ReleaseOptions,
            cancellationToken: cancellationToken);

        var modelDirectory = downloadResult.VariantDirectory;

        var configPath = Path.Combine(modelDirectory.FullName, "tm_config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Missing configuration file at '{configPath}'.");
        }

        var configSnapshot = await TableFormerConfigLoader.LoadAsync(
            configPath,
            modelDirectory.FullName,
            cancellationToken);

        return new TableFormerBootstrapResult(modelDirectory, configSnapshot, downloadResult.DownloadedFiles);
    }

    public void Dispose()
    {
        // No resources to dispose; retained for backwards compatibility with existing using patterns.
    }
}

public sealed record TableFormerBootstrapResult(
    DirectoryInfo ModelDirectory,
    TableFormerConfigSnapshot ConfigSnapshot,
    IReadOnlyList<string> DownloadedFiles)
{
    public void EnsureConfigMatches(TableFormerConfigReference reference)
    {
        ConfigSnapshot.EnsureMatches(reference);
    }

    public Task<TableFormerInitializationSnapshot> InitializePredictorAsync(
        CancellationToken cancellationToken = default)
    {
        return TableFormerPredictorInitializer.InitializeAsync(
            ConfigSnapshot,
            ModelDirectory,
            cancellationToken);
    }
}
