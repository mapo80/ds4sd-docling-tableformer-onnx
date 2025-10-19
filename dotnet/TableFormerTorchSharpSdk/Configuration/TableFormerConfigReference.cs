using System.IO;
using System.Text.Json;

namespace TableFormerTorchSharpSdk.Configuration;

public sealed record TableFormerConfigReference(
    string RepoId,
    string Revision,
    string Variant,
    string CanonicalJson,
    string Sha256)
{
    public static async Task<TableFormerConfigReference> LoadAsync(
        string referencePath,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(referencePath))
        {
            throw new ArgumentException("Reference path must not be null or empty.", nameof(referencePath));
        }

        if (!File.Exists(referencePath))
        {
            throw new FileNotFoundException($"Reference file not found at '{referencePath}'.", referencePath);
        }

        await using var stream = File.OpenRead(referencePath);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        var root = document.RootElement;

        return new TableFormerConfigReference(
            RepoId: root.GetProperty("repo_id").GetString() ?? throw new InvalidDataException("Missing 'repo_id' in reference."),
            Revision: root.GetProperty("revision").GetString() ?? throw new InvalidDataException("Missing 'revision' in reference."),
            Variant: root.GetProperty("variant").GetString() ?? throw new InvalidDataException("Missing 'variant' in reference."),
            CanonicalJson: root.GetProperty("canonical_json").GetString() ?? throw new InvalidDataException("Missing 'canonical_json' in reference."),
            Sha256: root.GetProperty("sha256").GetString() ?? throw new InvalidDataException("Missing 'sha256' in reference."));
    }
}
