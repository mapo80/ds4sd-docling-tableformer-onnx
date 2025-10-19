using System.Collections.ObjectModel;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Safetensors;
using TableFormerTorchSharpSdk.Utilities;

namespace TableFormerTorchSharpSdk.Initialization;

public static class TableFormerPredictorInitializer
{
    public static async Task<TableFormerInitializationSnapshot> InitializeAsync(
        TableFormerConfigSnapshot configSnapshot,
        DirectoryInfo modelDirectory,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(configSnapshot);
        ArgumentNullException.ThrowIfNull(modelDirectory);

        var wordMapObject = LoadWordMapObject(configSnapshot.Config, modelDirectory);
        using var document = JsonDocument.Parse(wordMapObject.ToJsonString());
        var canonicalJson = JsonCanonicalizer.GetCanonicalJson(document.RootElement);
        var sha256 = ComputeSha256(canonicalJson);

        var wordMapTag = ReadWordMapSection(wordMapObject, "word_map_tag");
        var reverseWordMapTag = BuildReverseMap(wordMapTag);
        var wordMapCell = ReadWordMapSection(wordMapObject, "word_map_cell");

        var wordMapSnapshot = new TableFormerWordMapSnapshot(
            new ReadOnlyDictionary<string, int>(wordMapTag),
            new ReadOnlyDictionary<int, string>(reverseWordMapTag),
            new ReadOnlyDictionary<string, int>(wordMapCell),
            canonicalJson,
            sha256);

        var safetensorsFiles = modelDirectory
            .GetFiles("*.safetensors", SearchOption.TopDirectoryOnly)
            .OrderBy(f => f.Name, StringComparer.Ordinal)
            .ToArray();

        if (safetensorsFiles.Length == 0)
        {
            throw new FileNotFoundException(
                $"No safetensors files were found in '{modelDirectory.FullName}'.");
        }

        var tensorDigests = new List<SafeTensorFileRecord>(safetensorsFiles.Length);
        foreach (var file in safetensorsFiles)
        {
            tensorDigests.Add(await SafeTensorFile.ComputeDigestsAsync(file, cancellationToken).ConfigureAwait(false));
        }

        return new TableFormerInitializationSnapshot(wordMapSnapshot, tensorDigests);
    }

    private static JsonObject LoadWordMapObject(JsonObject config, DirectoryInfo modelDirectory)
    {
        if (config.TryGetPropertyValue("dataset_wordmap", out var inlineWordMap) && inlineWordMap is JsonObject inlineObject)
        {
            return inlineObject;
        }

        if (!config.TryGetPropertyValue("dataset", out var datasetNode) || datasetNode is not JsonObject datasetObject)
        {
            throw new InvalidDataException(
                "Configuration does not contain 'dataset' details required to locate the word map.");
        }

        if (!datasetObject.TryGetPropertyValue("prepared_data_dir", out var preparedDirNode) || preparedDirNode is null)
        {
            throw new InvalidDataException(
                "Configuration is missing both 'dataset_wordmap' inline data and 'dataset.prepared_data_dir'.");
        }

        var preparedDir = preparedDirNode.GetValue<string>();
        if (string.IsNullOrWhiteSpace(preparedDir))
        {
            throw new InvalidDataException("Configuration specifies an empty 'dataset.prepared_data_dir'.");
        }

        if (!datasetObject.TryGetPropertyValue("name", out var datasetNameNode) || datasetNameNode is null)
        {
            throw new InvalidDataException("Configuration is missing 'dataset.name' required to resolve the word map file.");
        }

        var datasetName = datasetNameNode.GetValue<string>();
        var fileName = PreparedDataLayout.GetFileName("WORDMAP", datasetName);
        var resolvedDirectory = ResolvePreparedDataDirectory(preparedDir, modelDirectory);
        var fullPath = Path.Combine(resolvedDirectory, fileName);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException(
                $"Word map file '{fileName}' not found in '{resolvedDirectory}'.",
                fullPath);
        }

        var jsonText = File.ReadAllText(fullPath);
        return JsonNode.Parse(jsonText)?.AsObject()
            ?? throw new InvalidDataException(
                $"Word map file '{fullPath}' does not contain a JSON object.");
    }

    private static string ResolvePreparedDataDirectory(string preparedDir, DirectoryInfo modelDirectory)
    {
        if (Path.IsPathRooted(preparedDir))
        {
            return preparedDir;
        }

        // Mirror the Python implementation which joins the configured path with the provided directory.
        return Path.GetFullPath(Path.Combine(modelDirectory.FullName, preparedDir));
    }

    private static Dictionary<string, int> ReadWordMapSection(JsonObject wordMap, string sectionName)
    {
        if (!wordMap.TryGetPropertyValue(sectionName, out var sectionNode) || sectionNode is not JsonObject sectionObject)
        {
            throw new InvalidDataException($"Word map JSON is missing the '{sectionName}' section.");
        }

        var result = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var kvp in sectionObject)
        {
            if (kvp.Value is null)
            {
                throw new InvalidDataException(
                    $"Word map entry '{kvp.Key}' in '{sectionName}' has a null value.");
            }

            result[kvp.Key] = kvp.Value.GetValue<int>();
        }

        return result;
    }

    private static Dictionary<int, string> BuildReverseMap(Dictionary<string, int> forwardMap)
    {
        var reverse = new Dictionary<int, string>();
        foreach (var kvp in forwardMap)
        {
            if (!reverse.TryAdd(kvp.Value, kvp.Key))
            {
                throw new InvalidDataException(
                    $"Duplicate word map index '{kvp.Value}' assigned to '{reverse[kvp.Value]}' and '{kvp.Key}'.");
            }
        }

        return reverse;
    }

    private static string ComputeSha256(string canonicalJson)
    {
        var bytes = Encoding.UTF8.GetBytes(canonicalJson);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}

public sealed record TableFormerInitializationSnapshot(
    TableFormerWordMapSnapshot WordMap,
    IReadOnlyList<SafeTensorFileRecord> WeightFiles)
{
    public void EnsureMatches(TableFormerInitializationReference reference)
    {
        ArgumentNullException.ThrowIfNull(reference);

        WordMap.EnsureMatches(reference);
        EnsureTensorDigestsMatch(reference);
    }

    private void EnsureTensorDigestsMatch(TableFormerInitializationReference reference)
    {
        if (WeightFiles.Count != reference.WeightFiles.Count)
        {
            throw new InvalidDataException(
                $"Tensor file count mismatch. Expected {reference.WeightFiles.Count}, got {WeightFiles.Count}.");
        }

        var actualByName = WeightFiles.ToDictionary(f => f.FileName, StringComparer.Ordinal);
        foreach (var expected in reference.WeightFiles)
        {
            if (!actualByName.TryGetValue(expected.FileName, out var actual))
            {
                throw new InvalidDataException(
                    $"Missing safetensors file '{expected.FileName}' in computed digests.");
            }

            if (!string.Equals(actual.Sha256, expected.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"File SHA-256 mismatch for '{expected.FileName}'. Expected {expected.Sha256}, got {actual.Sha256}.");
            }

            EnsureTensorListMatches(expected, actual);
        }
    }

    private static void EnsureTensorListMatches(SafeTensorFileRecord expected, SafeTensorFileRecord actual)
    {
        if (actual.Tensors.Count != expected.Tensors.Count)
        {
            throw new InvalidDataException(
                $"Tensor count mismatch for '{expected.FileName}'. Expected {expected.Tensors.Count}, got {actual.Tensors.Count}.");
        }

        var actualByName = actual.Tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
        foreach (var expectedTensor in expected.Tensors)
        {
            if (!actualByName.TryGetValue(expectedTensor.Name, out var actualTensor))
            {
                throw new InvalidDataException(
                    $"Missing tensor '{expectedTensor.Name}' in file '{expected.FileName}'.");
            }

            if (!string.Equals(actualTensor.Dtype, expectedTensor.Dtype, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Tensor dtype mismatch for '{expectedTensor.Name}'. Expected {expectedTensor.Dtype}, got {actualTensor.Dtype}.");
            }

            if (!actualTensor.Shape.SequenceEqual(expectedTensor.Shape))
            {
                throw new InvalidDataException(
                    $"Tensor shape mismatch for '{expectedTensor.Name}'.");
            }

            if (!string.Equals(actualTensor.Sha256, expectedTensor.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"Tensor SHA-256 mismatch for '{expectedTensor.Name}'. Expected {expectedTensor.Sha256}, got {actualTensor.Sha256}.");
            }
        }
    }
}

public sealed record TableFormerWordMapSnapshot(
    IReadOnlyDictionary<string, int> TagMap,
    IReadOnlyDictionary<int, string> ReverseTagMap,
    IReadOnlyDictionary<string, int> CellMap,
    string CanonicalJson,
    string Sha256)
{
    public void EnsureMatches(TableFormerInitializationReference reference)
    {
        ArgumentNullException.ThrowIfNull(reference);

        if (!string.Equals(Sha256, reference.WordMapSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Word map SHA-256 mismatch. Expected {reference.WordMapSha256}, got {Sha256}.");
        }

        if (!string.Equals(CanonicalJson, reference.WordMapCanonicalJson, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Word map canonical JSON mismatch between Python reference and .NET snapshot.");
        }

        EnsureDictionaryMatches(TagMap, reference.WordMapTag, "word_map_tag");
        EnsureDictionaryMatches(CellMap, reference.WordMapCell, "word_map_cell");
    }

    private static void EnsureDictionaryMatches(
        IReadOnlyDictionary<string, int> actual,
        IReadOnlyDictionary<string, int> expected,
        string name)
    {
        if (actual.Count != expected.Count)
        {
            throw new InvalidDataException(
                $"Mismatch in number of entries for '{name}'. Expected {expected.Count}, got {actual.Count}.");
        }

        foreach (var kvp in expected)
        {
            if (!actual.TryGetValue(kvp.Key, out var value) || value != kvp.Value)
            {
                throw new InvalidDataException(
                    $"Value mismatch for key '{kvp.Key}' in '{name}'. Expected {kvp.Value}, got {value}.");
            }
        }
    }
}

internal static class PreparedDataLayout
{
    private static readonly IReadOnlyDictionary<string, string> Templates = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["BBOXES"] = "BBOXES.json",
        ["IMAGES"] = "IMAGES.json",
        ["STATISTICS"] = "STATISTICS_<POSTFIX>.json",
        ["TRAIN_CELLBBOXES"] = "TRAIN_CELLBBOXES_<POSTFIX>.json",
        ["TRAIN_CELLLENS"] = "TRAIN_CELLLENS_<POSTFIX>.json",
        ["TRAIN_CELLS"] = "TRAIN_CELLS_<POSTFIX>.json",
        ["TRAIN_TAGLENS"] = "TRAIN_TAGLENS_<POSTFIX>.json",
        ["TRAIN_TAGS"] = "TRAIN_TAGS_<POSTFIX>.json",
        ["VAL"] = "VAL.json",
        ["WORDMAP"] = "WORDMAP_<POSTFIX>.json",
    };

    public static string GetFileName(string part, string datasetName)
    {
        if (!Templates.TryGetValue(part, out var template))
        {
            throw new InvalidDataException($"Unknown prepared data part '{part}'.");
        }

        if (template.Contains("<POSTFIX>", StringComparison.Ordinal))
        {
            return template.Replace("<POSTFIX>", datasetName, StringComparison.Ordinal);
        }

        return template;
    }
}
