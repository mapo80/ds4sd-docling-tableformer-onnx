using System.Collections.ObjectModel;
using System.Linq;
using System.Text.Json;
using TableFormerTorchSharpSdk.Safetensors;

namespace TableFormerTorchSharpSdk.Initialization;

public sealed class TableFormerInitializationReference
{
    private TableFormerInitializationReference(
        IReadOnlyDictionary<string, int> wordMapTag,
        IReadOnlyDictionary<string, int> wordMapCell,
        string canonicalJson,
        string sha256,
        IReadOnlyList<SafeTensorFileRecord> weightFiles)
    {
        WordMapTag = wordMapTag;
        WordMapCell = wordMapCell;
        WordMapCanonicalJson = canonicalJson;
        WordMapSha256 = sha256;
        WeightFiles = weightFiles;
    }

    public IReadOnlyDictionary<string, int> WordMapTag { get; }

    public IReadOnlyDictionary<string, int> WordMapCell { get; }

    public string WordMapCanonicalJson { get; }

    public string WordMapSha256 { get; }

    public IReadOnlyList<SafeTensorFileRecord> WeightFiles { get; }

    public static async Task<TableFormerInitializationReference> LoadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        await using var stream = File.OpenRead(path);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
        var root = document.RootElement;

        var wordMapElement = RequireProperty(root, "word_map");
        var canonicalJson = RequireString(root, "word_map_canonical_json");
        var sha256 = RequireString(root, "word_map_sha256");

        var wordMapTag = ReadDictionary(wordMapElement, "word_map_tag");
        var wordMapCell = ReadDictionary(wordMapElement, "word_map_cell");

        var weightFiles = ReadWeightFiles(root.GetProperty("weight_files"));

        return new TableFormerInitializationReference(
            new ReadOnlyDictionary<string, int>(wordMapTag),
            new ReadOnlyDictionary<string, int>(wordMapCell),
            canonicalJson,
            sha256,
            weightFiles);
    }

    private static Dictionary<string, int> ReadDictionary(JsonElement parent, string propertyName)
    {
        if (!parent.TryGetProperty(propertyName, out var property))
        {
            throw new InvalidDataException($"Reference JSON is missing '{propertyName}'.");
        }

        var result = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var item in property.EnumerateObject())
        {
            result[item.Name] = item.Value.GetInt32();
        }

        return result;
    }

    private static IReadOnlyList<SafeTensorFileRecord> ReadWeightFiles(JsonElement array)
    {
        var result = new List<SafeTensorFileRecord>();
        foreach (var element in array.EnumerateArray())
        {
            var fileName = RequireString(element, "file_name");
            var sha256 = RequireString(element, "sha256");
            var tensors = ReadTensorArray(element.GetProperty("tensors"));
            result.Add(new SafeTensorFileRecord(fileName, sha256, tensors));
        }

        return result.AsReadOnly();
    }

    private static IReadOnlyList<SafeTensorTensorRecord> ReadTensorArray(JsonElement array)
    {
        var tensors = new List<SafeTensorTensorRecord>();
        foreach (var element in array.EnumerateArray())
        {
            var name = RequireString(element, "name");
            var dtype = RequireString(element, "dtype");
            var shape = element.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            var sha256 = RequireString(element, "sha256");
            tensors.Add(new SafeTensorTensorRecord(name, dtype, shape, sha256));
        }

        return tensors.AsReadOnly();
    }

    private static string RequireString(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var property) || property.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException($"Reference JSON is missing string property '{propertyName}'.");
        }

        return property.GetString()!;
    }

    private static JsonElement RequireProperty(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var property))
        {
            throw new InvalidDataException($"Reference JSON is missing '{propertyName}'.");
        }

        return property;
    }
}
