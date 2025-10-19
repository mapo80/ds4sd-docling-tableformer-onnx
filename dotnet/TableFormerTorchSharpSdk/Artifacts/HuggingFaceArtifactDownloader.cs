using System.Collections.Concurrent;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Text.Json;

namespace TableFormerTorchSharpSdk.Artifacts;

internal sealed class HuggingFaceArtifactDownloader
{
    private static readonly Uri ApiBaseUri = new("https://huggingface.co/");
    private readonly HttpClient _httpClient;

    public HuggingFaceArtifactDownloader(HttpClient httpClient)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
    }

    public async Task<IReadOnlyList<string>> DownloadArtifactsAsync(
        string repoId,
        string revision,
        string variant,
        DirectoryInfo targetDirectory,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);
        ArgumentException.ThrowIfNullOrWhiteSpace(variant);
        ArgumentNullException.ThrowIfNull(targetDirectory);

        targetDirectory.Create();

        var basePath = $"model_artifacts/tableformer/{variant}";
        var treeEntries = await GetTreeEntriesAsync(repoId, revision, basePath, cancellationToken);

        var filesToDownload = treeEntries
            .Where(entry => entry.Type == "file" && entry.Path.StartsWith(basePath, StringComparison.Ordinal))
            .Select(entry => entry.Path)
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        var downloadedFiles = new ConcurrentBag<string>();

        foreach (var filePath in filesToDownload)
        {
            var relativePath = filePath.Substring(basePath.Length).TrimStart('/');
            if (string.IsNullOrEmpty(relativePath))
            {
                continue;
            }

            var destinationPath = Path.Combine(targetDirectory.FullName, relativePath.Replace('/', Path.DirectorySeparatorChar));
            var destinationDirectory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(destinationDirectory))
            {
                Directory.CreateDirectory(destinationDirectory);
            }

            if (File.Exists(destinationPath))
            {
                continue;
            }

            await DownloadFileAsync(repoId, revision, filePath, destinationPath, downloadedFiles, cancellationToken);
        }

        return downloadedFiles.OrderBy(path => path, StringComparer.Ordinal).ToArray();
    }

    private async Task DownloadFileAsync(
        string repoId,
        string revision,
        string filePath,
        string destinationPath,
        ConcurrentBag<string> downloadedFiles,
        CancellationToken cancellationToken)
    {
        var resolvedRepoId = string.Join('/', repoId.Split('/', StringSplitOptions.RemoveEmptyEntries).Select(Uri.EscapeDataString));
        var resolvedPath = string.Join('/', filePath.Split('/', StringSplitOptions.RemoveEmptyEntries).Select(Uri.EscapeDataString));
        var downloadUri = new Uri(ApiBaseUri, $"{resolvedRepoId}/resolve/{Uri.EscapeDataString(revision)}/{resolvedPath}?download=1");

        using var response = await _httpClient.GetAsync(downloadUri, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);

        if (filePath.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            using var buffer = new MemoryStream();
            await contentStream.CopyToAsync(buffer, cancellationToken);
            buffer.Position = 0;

            using var archive = new ZipArchive(buffer, ZipArchiveMode.Read, leaveOpen: false);
            foreach (var entry in archive.Entries)
            {
                if (string.IsNullOrEmpty(entry.FullName))
                {
                    continue;
                }

                var entryDestination = Path.Combine(Path.GetDirectoryName(destinationPath) ?? string.Empty, entry.FullName.Replace('/', Path.DirectorySeparatorChar));
                var entryDirectory = Path.GetDirectoryName(entryDestination);
                if (!string.IsNullOrEmpty(entryDirectory))
                {
                    Directory.CreateDirectory(entryDirectory);
                }

                if (entry.FullName.EndsWith("/", StringComparison.Ordinal))
                {
                    continue;
                }

                await using var entryStream = entry.Open();
                await using var fileStream = File.Create(entryDestination);
                await entryStream.CopyToAsync(fileStream, cancellationToken);
                downloadedFiles.Add(entryDestination);
            }
        }
        else
        {
            await using var fileStream = File.Create(destinationPath);
            await contentStream.CopyToAsync(fileStream, cancellationToken);
            downloadedFiles.Add(destinationPath);
        }
    }

    private async Task<List<TreeEntry>> GetTreeEntriesAsync(
        string repoId,
        string revision,
        string basePath,
        CancellationToken cancellationToken)
    {
        var resolvedRepoId = string.Join('/', repoId.Split('/', StringSplitOptions.RemoveEmptyEntries).Select(Uri.EscapeDataString));
        var requestUri = new Uri(ApiBaseUri, $"api/models/{resolvedRepoId}/tree/{Uri.EscapeDataString(revision)}?recursive=1&path={Uri.EscapeDataString(basePath)}");

        using var response = await _httpClient.GetAsync(requestUri, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);

        var entries = new List<TreeEntry>();
        foreach (var element in document.RootElement.EnumerateArray())
        {
            if (!element.TryGetProperty("type", out var typeElement) || !element.TryGetProperty("path", out var pathElement))
            {
                continue;
            }

            var type = typeElement.GetString();
            var path = pathElement.GetString();
            if (string.IsNullOrEmpty(type) || string.IsNullOrEmpty(path))
            {
                continue;
            }

            entries.Add(new TreeEntry(type, path));
        }

        return entries;
    }

    private sealed record TreeEntry(string Type, string Path);
}
