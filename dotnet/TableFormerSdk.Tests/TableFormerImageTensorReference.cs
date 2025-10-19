using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace TableFormerSdk.Tests;

public sealed class TableFormerImageTensorReference
{
    private TableFormerImageTensorReference(
        int targetSize,
        int channels,
        IReadOnlyList<double> mean,
        IReadOnlyList<double> std,
        IReadOnlyList<TableFormerImageTensorSample> samples)
    {
        TargetSize = targetSize;
        Channels = channels;
        Mean = mean;
        Std = std;
        Samples = samples;
    }

    public int TargetSize { get; }

    public int Channels { get; }

    public IReadOnlyList<double> Mean { get; }

    public IReadOnlyList<double> Std { get; }

    public IReadOnlyList<TableFormerImageTensorSample> Samples { get; }

    public static TableFormerImageTensorReference Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Reference file not found at '{path}'.", path);
        }

        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        var targetSize = root.GetProperty("target_size").GetInt32();
        var channels = root.GetProperty("channels").GetInt32();

        var normalization = root.GetProperty("normalization");
        var mean = normalization.GetProperty("mean").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var std = normalization.GetProperty("std").EnumerateArray().Select(v => v.GetDouble()).ToArray();

        var samples = new List<TableFormerImageTensorSample>();
        foreach (var sampleElement in root.GetProperty("samples").EnumerateArray())
        {
            samples.Add(TableFormerImageTensorSample.FromJson(sampleElement));
        }

        return new TableFormerImageTensorReference(targetSize, channels, mean, std, samples);
    }
}

public sealed class TableFormerImageTensorSample
{
    private TableFormerImageTensorSample(
        string imageName,
        int tableIndex,
        IReadOnlyList<long> tensorShape,
        string tensorSha256,
        string tensorZlibBase64,
        double tensorMin,
        double tensorMax,
        double tensorMean,
        double tensorStd)
    {
        ImageName = imageName;
        TableIndex = tableIndex;
        TensorShape = tensorShape;
        TensorSha256 = tensorSha256;
        TensorZlibBase64 = tensorZlibBase64;
        TensorMin = tensorMin;
        TensorMax = tensorMax;
        TensorMean = tensorMean;
        TensorStd = tensorStd;
    }

    public string ImageName { get; }

    public int TableIndex { get; }

    public IReadOnlyList<long> TensorShape { get; }

    public string TensorSha256 { get; }

    public string TensorZlibBase64 { get; }

    public double TensorMin { get; }

    public double TensorMax { get; }

    public double TensorMean { get; }

    public double TensorStd { get; }

    public int TensorValueCount => (int)TensorShape.Aggregate(1L, (accumulator, dimension) => accumulator * dimension);

    public static TableFormerImageTensorSample FromJson(JsonElement element)
    {
        var imageName = element.GetProperty("image_name").GetString()
            ?? throw new InvalidDataException("Tensor reference is missing 'image_name'.");

        var tableIndex = element.GetProperty("table_index").GetInt32();
        var tensorShape = element.GetProperty("tensor_shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();

        var tensorSha256 = element.GetProperty("tensor_sha256").GetString()
            ?? throw new InvalidDataException("Tensor reference is missing 'tensor_sha256'.");

        var tensorZlibBase64 = element.GetProperty("tensor_zlib_base64").GetString()
            ?? throw new InvalidDataException("Tensor reference is missing 'tensor_zlib_base64'.");

        var tensorMin = element.GetProperty("tensor_min").GetDouble();
        var tensorMax = element.GetProperty("tensor_max").GetDouble();
        var tensorMean = element.GetProperty("tensor_mean").GetDouble();
        var tensorStd = element.GetProperty("tensor_std").GetDouble();

        return new TableFormerImageTensorSample(
            imageName,
            tableIndex,
            tensorShape,
            tensorSha256,
            tensorZlibBase64,
            tensorMin,
            tensorMax,
            tensorMean,
            tensorStd);
    }

    public float[] DecodeTensorValues()
    {
        var compressed = Convert.FromBase64String(TensorZlibBase64);
        using var input = new MemoryStream(compressed);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);

        var bytes = output.ToArray();
        if (bytes.Length % sizeof(float) != 0)
        {
            throw new InvalidDataException("Decompressed tensor bytes are not aligned to 32-bit floats.");
        }

        var floatCount = bytes.Length / sizeof(float);
        var result = new float[floatCount];
        MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).CopyTo(result);

        if (result.Length != TensorValueCount)
        {
            throw new InvalidDataException(
                $"Tensor length mismatch. Expected {TensorValueCount} floats but decoded {result.Length}.");
        }

        return result;
    }
}
