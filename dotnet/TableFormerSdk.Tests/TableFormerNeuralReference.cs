using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace TableFormerSdk.Tests;

public sealed class TableFormerNeuralReference
{
    private TableFormerNeuralReference(IReadOnlyList<TableFormerNeuralSample> samples)
    {
        Samples = samples;
    }

    public IReadOnlyList<TableFormerNeuralSample> Samples { get; }

    public static TableFormerNeuralReference Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        if (!root.TryGetProperty("samples", out var samplesElement))
        {
            throw new InvalidDataException("Neural reference JSON is missing the 'samples' array.");
        }

        var samples = new List<TableFormerNeuralSample>();
        foreach (var element in samplesElement.EnumerateArray())
        {
            samples.Add(TableFormerNeuralSample.FromJson(element));
        }

        return new TableFormerNeuralReference(samples.AsReadOnly());
    }
}

public sealed class TableFormerNeuralSample
{
    private TableFormerNeuralSample(
        string imageName,
        int tableIndex,
        string tensorSha256,
        IReadOnlyList<int> tagSequence,
        string tagSequenceSha256,
        IReadOnlyList<int> classShape,
        string classZlibBase64,
        string classSha256,
        double? classMin,
        double? classMax,
        double? classMean,
        double? classStd,
        IReadOnlyList<int> coordShape,
        string coordZlibBase64,
        string coordSha256,
        double? coordMin,
        double? coordMax,
        double? coordMean,
        double? coordStd)
    {
        ImageName = imageName;
        TableIndex = tableIndex;
        TensorSha256 = tensorSha256;
        TagSequence = tagSequence;
        TagSequenceSha256 = tagSequenceSha256;
        ClassShape = classShape;
        ClassZlibBase64 = classZlibBase64;
        ClassSha256 = classSha256;
        ClassMin = classMin;
        ClassMax = classMax;
        ClassMean = classMean;
        ClassStd = classStd;
        CoordShape = coordShape;
        CoordZlibBase64 = coordZlibBase64;
        CoordSha256 = coordSha256;
        CoordMin = coordMin;
        CoordMax = coordMax;
        CoordMean = coordMean;
        CoordStd = coordStd;
    }

    public string ImageName { get; }

    public int TableIndex { get; }

    public string TensorSha256 { get; }

    public IReadOnlyList<int> TagSequence { get; }

    public string TagSequenceSha256 { get; }

    public IReadOnlyList<int> ClassShape { get; }

    public string ClassZlibBase64 { get; }

    public string ClassSha256 { get; }

    public double? ClassMin { get; }

    public double? ClassMax { get; }

    public double? ClassMean { get; }

    public double? ClassStd { get; }

    public IReadOnlyList<int> CoordShape { get; }

    public string CoordZlibBase64 { get; }

    public string CoordSha256 { get; }

    public double? CoordMin { get; }

    public double? CoordMax { get; }

    public double? CoordMean { get; }

    public double? CoordStd { get; }

    public int ClassValueCount => ClassShape.Aggregate(1, (product, dimension) => product * dimension);

    public int CoordValueCount => CoordShape.Aggregate(1, (product, dimension) => product * dimension);

    public float[] DecodeClassValues() => DecodeFloatArray(ClassZlibBase64, ClassValueCount);

    public float[] DecodeCoordValues() => DecodeFloatArray(CoordZlibBase64, CoordValueCount);

    private static float[] DecodeFloatArray(string base64, int expectedCount)
    {
        var compressed = Convert.FromBase64String(base64);
        using var input = new MemoryStream(compressed);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);

        var bytes = output.ToArray();
        if (bytes.Length % sizeof(float) != 0)
        {
            throw new InvalidDataException("Decoded byte count is not aligned to 32-bit floats.");
        }

        var result = new float[bytes.Length / sizeof(float)];
        MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).CopyTo(result);

        if (expectedCount != result.Length)
        {
            throw new InvalidDataException(
                $"Decoded float count mismatch. Expected {expectedCount}, decoded {result.Length}.");
        }

        return result;
    }

    public static TableFormerNeuralSample FromJson(JsonElement element)
    {
        var imageName = element.GetProperty("image_name").GetString()
            ?? throw new InvalidDataException("Neural sample is missing 'image_name'.");
        var tableIndex = element.GetProperty("table_index").GetInt32();
        var tensorSha = element.GetProperty("tensor_sha256").GetString()
            ?? throw new InvalidDataException("Neural sample is missing 'tensor_sha256'.");

        var tagSequence = element.GetProperty("tag_sequence").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var tagSha = element.GetProperty("tag_sequence_sha256").GetString()
            ?? throw new InvalidDataException("Neural sample is missing 'tag_sequence_sha256'.");

        var classShape = element.GetProperty("class_shape").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var classZlib = element.GetProperty("class_zlib_base64").GetString()
            ?? throw new InvalidDataException("Neural sample is missing 'class_zlib_base64'.");
        var classSha = element.GetProperty("class_sha256").GetString()
            ?? throw new InvalidDataException("Neural sample is missing 'class_sha256'.");

        var classMin = GetNullableDouble(element, "class_min");
        var classMax = GetNullableDouble(element, "class_max");
        var classMean = GetNullableDouble(element, "class_mean");
        var classStd = GetNullableDouble(element, "class_std");

        var coordShape = element.GetProperty("coord_shape").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var coordZlib = element.GetProperty("coord_zlib_base64").GetString()
            ?? throw new InvalidDataException("Neural sample is missing 'coord_zlib_base64'.");
        var coordSha = element.GetProperty("coord_sha256").GetString()
            ?? throw new InvalidDataException("Neural sample is missing 'coord_sha256'.");

        var coordMin = GetNullableDouble(element, "coord_min");
        var coordMax = GetNullableDouble(element, "coord_max");
        var coordMean = GetNullableDouble(element, "coord_mean");
        var coordStd = GetNullableDouble(element, "coord_std");

        return new TableFormerNeuralSample(
            imageName,
            tableIndex,
            tensorSha,
            tagSequence.AsReadOnly(),
            tagSha,
            classShape.AsReadOnly(),
            classZlib,
            classSha,
            classMin,
            classMax,
            classMean,
            classStd,
            coordShape.AsReadOnly(),
            coordZlib,
            coordSha,
            coordMin,
            coordMax,
            coordMean,
            coordStd);
    }

    private static double? GetNullableDouble(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var property) || property.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return property.GetDouble();
    }
}
