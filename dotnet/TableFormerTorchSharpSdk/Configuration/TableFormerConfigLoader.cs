using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

using TableFormerTorchSharpSdk.Utilities;

namespace TableFormerTorchSharpSdk.Configuration;

public static class TableFormerConfigLoader
{
    public static async Task<TableFormerConfigSnapshot> LoadAsync(
        string configPath,
        string modelDirectory,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(configPath))
        {
            throw new ArgumentException("Config path must not be null or empty.", nameof(configPath));
        }

        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            throw new ArgumentException("Model directory must not be null or empty.", nameof(modelDirectory));
        }

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Configuration file not found at '{configPath}'.", configPath);
        }

        await using var stream = File.OpenRead(configPath);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);

        ValidateConfig(document.RootElement);

        var normalizedJson = JsonCanonicalizer.GetCanonicalJson(document.RootElement);
        var sha256 = ComputeSha256(normalizedJson);

        stream.Position = 0;
        using var reader = new StreamReader(stream, Encoding.UTF8, leaveOpen: true);
        var jsonText = await reader.ReadToEndAsync(cancellationToken);

        var rootNode = JsonNode.Parse(jsonText)?.AsObject()
            ?? throw new InvalidDataException("The configuration file does not contain a JSON object at the root.");

        var normalizedModelDirectory = Path.GetFullPath(modelDirectory);
        EnsureModelSection(rootNode)["save_dir"] = normalizedModelDirectory;

        return new TableFormerConfigSnapshot(rootNode, normalizedJson, sha256);
    }

    private static void ValidateConfig(JsonElement root)
    {
        if (!root.TryGetProperty("model", out var modelElement))
        {
            return;
        }

        if (!root.TryGetProperty("preparation", out var preparationElement))
        {
            return;
        }

        if (!preparationElement.TryGetProperty("max_tag_len", out var maxTagLenElement))
        {
            throw new InvalidDataException("Config error: 'preparation.max_tag_len' parameter is missing");
        }

        if (!modelElement.TryGetProperty("seq_len", out var seqLenElement))
        {
            return;
        }

        var seqLen = GetPositiveInt(seqLenElement, "model.seq_len");
        var maxTagLen = GetPositiveInt(maxTagLenElement, "preparation.max_tag_len");

        if (seqLen <= 0)
        {
            throw new InvalidDataException("Config error: 'model.seq_len' should be positive");
        }

        if (seqLen > maxTagLen + 2)
        {
            throw new InvalidDataException("Config error: 'model.seq_len' should be up to 'preparation.max_tag_len' + 2");
        }
    }

    private static int GetPositiveInt(JsonElement element, string parameterName)
    {
        if (element.ValueKind == JsonValueKind.Number)
        {
            if (element.TryGetInt32(out var int32Value))
            {
                return int32Value;
            }

            if (element.TryGetInt64(out var int64Value))
            {
                if (int64Value is < int.MinValue or > int.MaxValue)
                {
                    throw new InvalidDataException($"Config error: '{parameterName}' is outside the supported range for 32-bit integers.");
                }

                return (int)int64Value;
            }

            if (element.TryGetDouble(out var doubleValue))
            {
                if (!double.IsFinite(doubleValue))
                {
                    throw new InvalidDataException($"Config error: '{parameterName}' must be a finite numeric value.");
                }

                return Convert.ToInt32(Math.Round(doubleValue), CultureInfo.InvariantCulture);
            }
        }

        throw new InvalidDataException($"Config error: '{parameterName}' should be numeric.");
    }

    private static JsonObject EnsureModelSection(JsonObject root)
    {
        if (root["model"] is JsonObject modelObject)
        {
            return modelObject;
        }

        modelObject = new JsonObject();
        root["model"] = modelObject;
        return modelObject;
    }

    private static string ComputeSha256(string normalizedJson)
    {
        var bytes = Encoding.UTF8.GetBytes(normalizedJson);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}

public sealed record TableFormerConfigSnapshot(JsonObject Config, string NormalizedJson, string Sha256Hash)
{
    public void EnsureMatches(TableFormerConfigReference reference)
    {
        ArgumentNullException.ThrowIfNull(reference);

        if (!string.Equals(Sha256Hash, reference.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Configuration SHA-256 hash mismatch between Python reference and .NET loader.");
        }

        if (!string.Equals(NormalizedJson, reference.CanonicalJson, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Normalized configuration JSON mismatch between Python reference and .NET loader.");
        }
    }
}
