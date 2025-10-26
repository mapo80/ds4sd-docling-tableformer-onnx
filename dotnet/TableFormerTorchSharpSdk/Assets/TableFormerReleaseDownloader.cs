using System.Collections.Concurrent;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using TableFormerTorchSharpSdk.Configuration;

namespace TableFormerTorchSharpSdk.Assets;

/// <summary>
/// Downloads TableFormer safetensor assets from a GitHub release on demand.
/// </summary>
public static class TableFormerReleaseDownloader
{
    private sealed record ReleaseAsset(
        string VariantName,
        string AssetName,
        string ZipRootPrefix,
        string Sha256,
        IReadOnlyList<string> RequiredRelativeFiles)
    {
        public string BuildUrl(TableFormerGithubReleaseOptions options) => $"{options.BuildBaseUrl().TrimEnd('/')}/{AssetName}";
    }

    private static readonly IReadOnlyDictionary<TableFormerModelVariant, ReleaseAsset> VariantAssets =
        new Dictionary<TableFormerModelVariant, ReleaseAsset>
        {
            [TableFormerModelVariant.Accurate] = new ReleaseAsset(
                VariantName: "accurate",
                AssetName: "tableformer-accurate.zip",
                ZipRootPrefix: "model_artifacts/tableformer/accurate/",
                Sha256: "3A316C01EFE1D70E0B8142ADB9FDA300F503ACF5F839CBF714DFF476C13959FE",
                RequiredRelativeFiles: new[]
                {
                    "tableformer_accurate.safetensors",
                    "tm_config.json"
                }),
            [TableFormerModelVariant.Fast] = new ReleaseAsset(
                VariantName: "fast",
                AssetName: "tableformer-fast.zip",
                ZipRootPrefix: "model_artifacts/tableformer/fast/",
                Sha256: "2E065D218BD0F42BDD4392D656463E4AD1E4F2220599D1B28DBEB0E84C0F5403",
                RequiredRelativeFiles: new[]
                {
                    "tableformer_fast.safetensors",
                    "tm_config.json"
                })
        };

    private static readonly HttpClient Http;
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> Locks = new(StringComparer.OrdinalIgnoreCase);

    static TableFormerReleaseDownloader()
    {
        Http = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        Http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("TableFormerTorchSharpSdk", "1.0"));
    }

    /// <summary>
    /// Ensure that the requested variant is available under the provided artifact root, downloading it when missing.
    /// </summary>
    public static async Task<TableFormerReleaseDownloadResult> EnsureVariantAsync(
        TableFormerModelVariant variant,
        DirectoryInfo artifactsRoot,
        TableFormerGithubReleaseOptions? releaseOptions = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(artifactsRoot);
        artifactsRoot.Create();

        if (!VariantAssets.TryGetValue(variant, out var asset))
        {
            throw new NotSupportedException($"Variant {variant} is not supported.");
        }

        var variantDirectory = GetVariantDirectory(artifactsRoot, asset.VariantName);
        var semaphore = Locks.GetOrAdd(variantDirectory.FullName, _ => new SemaphoreSlim(1, 1));
        await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            if (HasAllRequiredFiles(variantDirectory, asset.RequiredRelativeFiles))
            {
                logger?.Invoke($"✓ Modello {asset.VariantName} già presente: {variantDirectory.FullName}");
                return new TableFormerReleaseDownloadResult(variantDirectory, Array.Empty<string>());
            }

            var options = releaseOptions ?? new TableFormerGithubReleaseOptions();
            Directory.CreateDirectory(variantDirectory.FullName);

            var downloadPath = Path.Combine(Path.GetTempPath(), $"tableformer-{asset.VariantName}-{Guid.NewGuid():N}.zip");
            try
            {
                var assetUrl = asset.BuildUrl(options);
                logger?.Invoke($"↓ Download {asset.AssetName} da {assetUrl}");

                await DownloadAsync(assetUrl, downloadPath, cancellationToken).ConfigureAwait(false);

                VerifySha256(downloadPath, asset.Sha256);

                var extractedFiles = ExtractArchive(downloadPath, variantDirectory, asset.ZipRootPrefix, logger, cancellationToken);

                if (!HasAllRequiredFiles(variantDirectory, asset.RequiredRelativeFiles))
                {
                    throw new InvalidOperationException($"Dopo l'estrazione mancano alcuni file richiesti per la variante {asset.VariantName}.");
                }

                logger?.Invoke($"✓ Modello {asset.VariantName} disponibile in {variantDirectory.FullName}");
                return new TableFormerReleaseDownloadResult(variantDirectory, extractedFiles);
            }
            finally
            {
                TryDelete(downloadPath);
            }
        }
        finally
        {
            semaphore.Release();
        }
    }

    private static DirectoryInfo GetVariantDirectory(DirectoryInfo artifactsRoot, string variantName)
    {
        var variantPath = Path.Combine(
            artifactsRoot.FullName,
            "model_artifacts",
            "tableformer",
            variantName);

        var directory = new DirectoryInfo(variantPath);
        directory.Create();
        return directory;
    }

    private static bool HasAllRequiredFiles(DirectoryInfo directory, IReadOnlyList<string> requiredFiles)
    {
        return requiredFiles.All(file => File.Exists(Path.Combine(directory.FullName, file)));
    }

    private static async Task DownloadAsync(string url, string destinationPath, CancellationToken cancellationToken)
    {
        using var response = await Http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        await using var target = File.Open(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);
        await response.Content.CopyToAsync(target, cancellationToken).ConfigureAwait(false);
    }

    private static IReadOnlyList<string> ExtractArchive(
        string zipPath,
        DirectoryInfo destinationDirectory,
        string zipRootPrefix,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        using var archive = ZipFile.OpenRead(zipPath);
        var downloaded = new List<string>();

        foreach (var entry in archive.Entries)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (entry.FullName.EndsWith("/", StringComparison.Ordinal))
            {
                continue;
            }

            if (!entry.FullName.StartsWith(zipRootPrefix, StringComparison.Ordinal))
            {
                continue;
            }

            var relativePath = entry.FullName.Substring(zipRootPrefix.Length);
            var destinationPath = Path.Combine(destinationDirectory.FullName, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);

            logger?.Invoke($"→ Estrai {relativePath}");

            using var entryStream = entry.Open();
            using var fileStream = File.Create(destinationPath);
            entryStream.CopyTo(fileStream);
            downloaded.Add(destinationPath);
        }

        return downloaded;
    }

    private static void VerifySha256(string filePath, string expectedSha256)
    {
        using var stream = File.OpenRead(filePath);
        var hash = SHA256.HashData(stream);
        var builder = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            builder.Append(b.ToString("X2"));
        }

        if (!builder.ToString().Equals(expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"SHA-256 mismatch for '{Path.GetFileName(filePath)}'. Expected {expectedSha256}, got {builder}.");
        }
    }

    private static void TryDelete(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }

        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Ignore cleanup failures.
        }
    }
}

public sealed record TableFormerReleaseDownloadResult(DirectoryInfo VariantDirectory, IReadOnlyList<string> DownloadedFiles);
