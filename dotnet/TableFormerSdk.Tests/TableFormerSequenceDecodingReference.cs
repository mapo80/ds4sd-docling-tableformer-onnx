using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace TableFormerSdk.Tests;

public sealed class TableFormerSequenceDecodingReference
{
    private TableFormerSequenceDecodingReference(IReadOnlyList<TableFormerSequenceDecodingSample> samples)
    {
        Samples = samples;
    }

    public IReadOnlyList<TableFormerSequenceDecodingSample> Samples { get; }

    public static TableFormerSequenceDecodingReference Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        if (!root.TryGetProperty("samples", out var samplesElement))
        {
            throw new InvalidDataException("Sequence decoding reference JSON is missing the 'samples' array.");
        }

        var samples = new List<TableFormerSequenceDecodingSample>();
        foreach (var element in samplesElement.EnumerateArray())
        {
            samples.Add(TableFormerSequenceDecodingSample.FromJson(element));
        }

        return new TableFormerSequenceDecodingReference(samples.AsReadOnly());
    }
}

public sealed class TableFormerSequenceDecodingSample
{
    private TableFormerSequenceDecodingSample(
        string imageName,
        int tableIndex,
        string tensorSha256,
        string tagSequenceSha256,
        IReadOnlyList<int> tagSequence,
        IReadOnlyList<string> rsSequence,
        string rsSequenceSha256,
        IReadOnlyList<string> htmlSequence,
        string htmlSequenceSha256,
        IReadOnlyList<int> rawShape,
        string rawBase64,
        string rawSha256,
        double? rawMin,
        double? rawMax,
        double? rawMean,
        double? rawStd,
        IReadOnlyList<int> finalShape,
        string finalBase64,
        string finalSha256,
        double? finalMin,
        double? finalMax,
        double? finalMean,
        double? finalStd,
        bool bboxSync)
    {
        ImageName = imageName;
        TableIndex = tableIndex;
        TensorSha256 = tensorSha256;
        TagSequenceSha256 = tagSequenceSha256;
        TagSequence = tagSequence;
        RsSequence = rsSequence;
        RsSequenceSha256 = rsSequenceSha256;
        HtmlSequence = htmlSequence;
        HtmlSequenceSha256 = htmlSequenceSha256;
        RawShape = rawShape;
        RawBase64 = rawBase64;
        RawSha256 = rawSha256;
        RawMin = rawMin;
        RawMax = rawMax;
        RawMean = rawMean;
        RawStd = rawStd;
        FinalShape = finalShape;
        FinalBase64 = finalBase64;
        FinalSha256 = finalSha256;
        FinalMin = finalMin;
        FinalMax = finalMax;
        FinalMean = finalMean;
        FinalStd = finalStd;
        BoundingBoxesSynced = bboxSync;
    }

    public string ImageName { get; }

    public int TableIndex { get; }

    public string TensorSha256 { get; }

    public string TagSequenceSha256 { get; }

    public IReadOnlyList<int> TagSequence { get; }

    public IReadOnlyList<string> RsSequence { get; }

    public string RsSequenceSha256 { get; }

    public IReadOnlyList<string> HtmlSequence { get; }

    public string HtmlSequenceSha256 { get; }

    public IReadOnlyList<int> RawShape { get; }

    public string RawBase64 { get; }

    public string RawSha256 { get; }

    public double? RawMin { get; }

    public double? RawMax { get; }

    public double? RawMean { get; }

    public double? RawStd { get; }

    public IReadOnlyList<int> FinalShape { get; }

    public string FinalBase64 { get; }

    public string FinalSha256 { get; }

    public double? FinalMin { get; }

    public double? FinalMax { get; }

    public double? FinalMean { get; }

    public double? FinalStd { get; }

    public bool BoundingBoxesSynced { get; }

    public float[] DecodeRawBoundingBoxes() => DecodeFloatArray(RawBase64, RawShape);

    public float[] DecodeFinalBoundingBoxes() => DecodeFloatArray(FinalBase64, FinalShape);

    private static float[] DecodeFloatArray(string base64, IReadOnlyList<int> shape)
    {
        var expectedCount = shape.Aggregate(1, (product, dimension) => product * dimension);
        var compressed = Convert.FromBase64String(base64);
        using var input = new MemoryStream(compressed);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);

        var bytes = output.ToArray();
        if (bytes.Length % sizeof(float) != 0)
        {
            throw new InvalidDataException("Decoded bounding box byte count is not aligned to 32-bit floats.");
        }

        var result = new float[bytes.Length / sizeof(float)];
        MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).CopyTo(result);

        if (expectedCount != result.Length)
        {
            throw new InvalidDataException(
                $"Bounding box float count mismatch. Expected {expectedCount}, decoded {result.Length}.");
        }

        return result;
    }

    public static TableFormerSequenceDecodingSample FromJson(JsonElement element)
    {
        var imageName = element.GetProperty("image_name").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'image_name'.");
        var tableIndex = element.GetProperty("table_index").GetInt32();
        var tensorSha = element.GetProperty("tensor_sha256").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'tensor_sha256'.");
        var tagSha = element.GetProperty("tag_sequence_sha256").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'tag_sequence_sha256'.");

        var tagSequence = element.GetProperty("tag_sequence").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var rsSequence = element.GetProperty("rs_sequence").EnumerateArray().Select(e => e.GetString() ?? string.Empty).ToList();
        var rsSha = element.GetProperty("rs_sequence_sha256").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'rs_sequence_sha256'.");

        var htmlSequence = element.GetProperty("html_sequence").EnumerateArray().Select(e => e.GetString() ?? string.Empty).ToList();
        var htmlSha = element.GetProperty("html_sequence_sha256").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'html_sequence_sha256'.");

        var rawShape = element.GetProperty("raw_bbox_shape").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var rawBase64 = element.GetProperty("raw_bbox_zlib_base64").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'raw_bbox_zlib_base64'.");
        var rawSha = element.GetProperty("raw_bbox_sha256").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'raw_bbox_sha256'.");

        var rawMin = GetNullableDouble(element, "raw_bbox_min");
        var rawMax = GetNullableDouble(element, "raw_bbox_max");
        var rawMean = GetNullableDouble(element, "raw_bbox_mean");
        var rawStd = GetNullableDouble(element, "raw_bbox_std");

        var finalShape = element.GetProperty("final_bbox_shape").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var finalBase64 = element.GetProperty("final_bbox_zlib_base64").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'final_bbox_zlib_base64'.");
        var finalSha = element.GetProperty("final_bbox_sha256").GetString()
            ?? throw new InvalidDataException("Sequence sample is missing 'final_bbox_sha256'.");

        var finalMin = GetNullableDouble(element, "final_bbox_min");
        var finalMax = GetNullableDouble(element, "final_bbox_max");
        var finalMean = GetNullableDouble(element, "final_bbox_mean");
        var finalStd = GetNullableDouble(element, "final_bbox_std");

        var bboxSync = element.GetProperty("bbox_sync").GetBoolean();

        return new TableFormerSequenceDecodingSample(
            imageName,
            tableIndex,
            tensorSha,
            tagSha,
            tagSequence.AsReadOnly(),
            rsSequence.AsReadOnly(),
            rsSha,
            htmlSequence.AsReadOnly(),
            htmlSha,
            rawShape.AsReadOnly(),
            rawBase64,
            rawSha,
            rawMin,
            rawMax,
            rawMean,
            rawStd,
            finalShape.AsReadOnly(),
            finalBase64,
            finalSha,
            finalMin,
            finalMax,
            finalMean,
            finalStd,
            bboxSync);
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
