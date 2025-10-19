using System.Buffers.Binary;
using System.Linq;
using System.Text.Json;

using TorchSharp;

namespace TableFormerTorchSharpSdk.Safetensors;

internal sealed class SafeTensorLoader
{
    private readonly Dictionary<string, SafeTensorEntry> _entries = new(StringComparer.Ordinal);

    private SafeTensorLoader()
    {
    }

    public static SafeTensorLoader LoadFromFiles(IEnumerable<FileInfo> files)
    {
        ArgumentNullException.ThrowIfNull(files);

        var loader = new SafeTensorLoader();
        foreach (var file in files)
        {
            loader.LoadFile(file);
        }

        return loader;
    }

    public bool TryGetEntry(string name, out SafeTensorEntry entry) => _entries.TryGetValue(name, out entry!);

    public SafeTensorEntry this[string name] => _entries[name];

    public IEnumerable<SafeTensorEntry> Entries => _entries.Values;

    private void LoadFile(FileInfo file)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (!file.Exists)
        {
            throw new FileNotFoundException($"Safetensors file '{file.FullName}' was not found.", file.FullName);
        }

        using var stream = file.Open(FileMode.Open, FileAccess.Read, FileShare.Read);
        var headerSize = ReadHeaderSize(stream);
        var headerBytes = new byte[headerSize];
        stream.ReadExactly(headerBytes);

        using var headerDocument = JsonDocument.Parse(headerBytes);
        var tensorMetadata = ParseHeader(headerDocument.RootElement);
        var dataOffset = stream.Position;

        foreach (var metadata in tensorMetadata)
        {
            if (_entries.ContainsKey(metadata.Name))
            {
                throw new InvalidDataException($"Duplicate tensor '{metadata.Name}' detected while loading '{file.FullName}'.");
            }

            var entry = ReadTensor(stream, dataOffset, metadata);
            _entries[entry.Name] = entry;
        }
    }

    private static SafeTensorEntry ReadTensor(Stream stream, long dataOffset, SafeTensorMetadata metadata)
    {
        stream.Seek(dataOffset + metadata.Offset, SeekOrigin.Begin);
        var byteLength = checked((int)metadata.Length);
        var buffer = new byte[byteLength];
        stream.ReadExactly(buffer);

        return metadata.DType switch
        {
            "float32" or "F32" => CreateFloatEntry(metadata, buffer),
            "float64" or "F64" => CreateDoubleEntry(metadata, buffer),
            "int64" or "I64" => CreateLongEntry(metadata, buffer),
            "int32" or "I32" => CreateIntEntry(metadata, buffer),
            _ => throw new NotSupportedException($"Unsupported safetensors dtype '{metadata.DType}' for tensor '{metadata.Name}'.")
        };
    }

    private static SafeTensorEntry CreateFloatEntry(SafeTensorMetadata metadata, byte[] buffer)
    {
        var values = new float[metadata.ElementCount];
        var span = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(buffer.AsSpan());
        span.CopyTo(values);
        return new SafeTensorEntry(metadata.Name, "float32", metadata.Shape, values);
    }

    private static SafeTensorEntry CreateDoubleEntry(SafeTensorMetadata metadata, byte[] buffer)
    {
        var values = new double[metadata.ElementCount];
        var span = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(buffer.AsSpan());
        span.CopyTo(values);
        return new SafeTensorEntry(metadata.Name, "float64", metadata.Shape, values);
    }

    private static SafeTensorEntry CreateLongEntry(SafeTensorMetadata metadata, byte[] buffer)
    {
        var values = new long[metadata.ElementCount];
        var span = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, long>(buffer.AsSpan());
        span.CopyTo(values);
        return new SafeTensorEntry(metadata.Name, "int64", metadata.Shape, values);
    }

    private static SafeTensorEntry CreateIntEntry(SafeTensorMetadata metadata, byte[] buffer)
    {
        var values = new int[metadata.ElementCount];
        var span = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, int>(buffer.AsSpan());
        span.CopyTo(values);
        return new SafeTensorEntry(metadata.Name, "int32", metadata.Shape, values);
    }

    private static long ReadHeaderSize(Stream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(long)];
        if (stream.Read(buffer) != sizeof(long))
        {
            throw new EndOfStreamException("Unexpected end of file while reading safetensors header size.");
        }

        var headerSize = BinaryPrimitives.ReadInt64LittleEndian(buffer);
        if (headerSize < 0)
        {
            throw new InvalidDataException($"Safetensors header size cannot be negative (got {headerSize}).");
        }

        return headerSize;
    }

    private static IEnumerable<SafeTensorMetadata> ParseHeader(JsonElement root)
    {
        foreach (var property in root.EnumerateObject())
        {
            var dtype = property.Value.GetProperty("dtype").GetString()
                ?? throw new InvalidDataException($"Tensor '{property.Name}' is missing a dtype.");

            var offsets = property.Value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            if (offsets.Length != 2)
            {
                throw new InvalidDataException($"Tensor '{property.Name}' must specify exactly two data offsets.");
            }

            var start = offsets[0];
            var end = offsets[1];
            if (end < start)
            {
                throw new InvalidDataException($"Tensor '{property.Name}' has decreasing data offsets.");
            }

            var shape = property.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            yield return new SafeTensorMetadata(property.Name, NormalizeDType(dtype), shape, start, end - start);
        }
    }

    private static string NormalizeDType(string dtype)
    {
        return dtype switch
        {
            "F32" => "float32",
            "F64" => "float64",
            "BF16" => "bfloat16",
            "F16" => "float16",
            "I64" => "int64",
            "I32" => "int32",
            "I16" => "int16",
            "I8" => "int8",
            "U8" => "uint8",
            _ => dtype,
        };
    }

    private sealed record SafeTensorMetadata(string Name, string DType, long[] Shape, long Offset, long Length)
    {
        public int ElementCount => Shape.Length == 0 ? 1 : checked((int)Shape.Aggregate(1L, (current, dim) => current * dim));
    }
}

internal sealed class SafeTensorEntry
{
    private readonly float[]? _floatValues;
    private readonly double[]? _doubleValues;
    private readonly long[]? _longValues;
    private readonly int[]? _intValues;

    public SafeTensorEntry(string name, string dtype, long[] shape, float[] values)
    {
        Name = name;
        DType = dtype;
        Shape = shape;
        _floatValues = values;
    }

    public SafeTensorEntry(string name, string dtype, long[] shape, double[] values)
    {
        Name = name;
        DType = dtype;
        Shape = shape;
        _doubleValues = values;
    }

    public SafeTensorEntry(string name, string dtype, long[] shape, long[] values)
    {
        Name = name;
        DType = dtype;
        Shape = shape;
        _longValues = values;
    }

    public SafeTensorEntry(string name, string dtype, long[] shape, int[] values)
    {
        Name = name;
        DType = dtype;
        Shape = shape;
        _intValues = values;
    }

    public string Name { get; }

    public string DType { get; }

    public long[] Shape { get; }

    public torch.Tensor CreateTensor()
    {
        return DType switch
        {
            "float32" => torch.tensor(_floatValues ?? throw MissingValues(), Shape, dtype: torch.float32),
            "float64" => torch.tensor(_doubleValues ?? throw MissingValues(), Shape, dtype: torch.float64),
            "int64" => torch.tensor(_longValues ?? throw MissingValues(), Shape, dtype: torch.int64),
            "int32" => torch.tensor(_intValues ?? throw MissingValues(), Shape, dtype: torch.int32),
            _ => throw new NotSupportedException($"Unsupported dtype '{DType}' for tensor '{Name}'."),
        };
    }

    private Exception MissingValues()
    {
        return new InvalidOperationException($"Tensor '{Name}' of type '{DType}' does not contain loaded values.");
    }
}
