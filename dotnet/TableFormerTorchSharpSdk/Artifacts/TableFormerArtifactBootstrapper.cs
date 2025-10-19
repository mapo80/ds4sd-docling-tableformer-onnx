using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;

using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Initialization;

namespace TableFormerTorchSharpSdk.Artifacts;

public sealed class TableFormerArtifactBootstrapper : IDisposable
{
    private readonly bool _disposeHttpClient;
    private readonly HuggingFaceArtifactDownloader _downloader;
    private readonly HttpClient _httpClient;

    public TableFormerArtifactBootstrapper(
        DirectoryInfo artifactsRoot,
        string repoId = "ds4sd/docling-models",
        string revision = "main",
        string variant = "fast",
        HttpClient? httpClient = null)
    {
        ArgumentNullException.ThrowIfNull(artifactsRoot);
        ArgumentException.ThrowIfNullOrEmpty(repoId);
        ArgumentException.ThrowIfNullOrEmpty(revision);
        ArgumentException.ThrowIfNullOrEmpty(variant);

        ArtifactsRoot = artifactsRoot;
        RepoId = repoId;
        Revision = revision;
        Variant = variant;

        if (httpClient is null)
        {
            _httpClient = new HttpClient();
            _disposeHttpClient = true;
        }
        else
        {
            _httpClient = httpClient;
        }

        EnsureUserAgent(_httpClient.DefaultRequestHeaders);
        _downloader = new HuggingFaceArtifactDownloader(_httpClient);
    }

    public DirectoryInfo ArtifactsRoot { get; }

    public string RepoId { get; }

    public string Revision { get; }

    public string Variant { get; }

    public async Task<TableFormerBootstrapResult> EnsureArtifactsAsync(CancellationToken cancellationToken = default)
    {
        var modelDirectory = GetModelDirectory();
        var downloadedFiles = await _downloader.DownloadArtifactsAsync(
            RepoId,
            Revision,
            Variant,
            modelDirectory,
            cancellationToken);

        var configPath = Path.Combine(modelDirectory.FullName, "tm_config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Missing configuration file at '{configPath}'.");
        }

        var configSnapshot = await TableFormerConfigLoader.LoadAsync(
            configPath,
            modelDirectory.FullName,
            cancellationToken);

        return new TableFormerBootstrapResult(modelDirectory, configSnapshot, downloadedFiles);
    }

    public void Dispose()
    {
        if (_disposeHttpClient)
        {
            _httpClient.Dispose();
        }
    }

    private DirectoryInfo GetModelDirectory()
    {
        var variantDirectory = Path.Combine(
            ArtifactsRoot.FullName,
            "model_artifacts",
            "tableformer",
            Variant);

        var directoryInfo = new DirectoryInfo(variantDirectory);
        directoryInfo.Create();
        return directoryInfo;
    }

    private static void EnsureUserAgent(HttpRequestHeaders headers)
    {
        if (!headers.UserAgent.Any())
        {
            headers.UserAgent.Add(new ProductInfoHeaderValue("TableFormerTorchSharpSdk", "0.1"));
        }
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
