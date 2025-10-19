using System.Buffers;
using System.Buffers.Binary;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;

namespace TableFormerTorchSharpSdk.Safetensors;

public static class SafeTensorFile
{
    public static async Task<SafeTensorFileRecord> ComputeDigestsAsync(
        FileInfo file,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(file);

        await using var stream = file.Open(FileMode.Open, FileAccess.Read, FileShare.Read);
        var headerSize = await ReadHeaderSizeAsync(stream, cancellationToken).ConfigureAwait(false);
        var headerBytes = new byte[headerSize];
        await stream.ReadExactlyAsync(headerBytes.AsMemory(), cancellationToken).ConfigureAwait(false);

        using var headerDocument = JsonDocument.Parse(headerBytes);
        var dataOffset = stream.Position;
        var tensors = ParseTensorMetadata(headerDocument.RootElement);

        var tensorDigests = new List<SafeTensorTensorRecord>(tensors.Count);
        foreach (var tensor in tensors)
        {
            var sha256 = await ComputeTensorSha256Async(stream, dataOffset + tensor.OffsetStart, tensor.Length, cancellationToken)
                .ConfigureAwait(false);
            tensorDigests.Add(new SafeTensorTensorRecord(tensor.Name, NormalizeDtype(tensor.Dtype), tensor.Shape, sha256));
        }

        var fileSha = await ComputeFileSha256Async(file.FullName, cancellationToken).ConfigureAwait(false);
        return new SafeTensorFileRecord(file.Name, fileSha, tensorDigests.AsReadOnly());
    }

    private static async Task<long> ReadHeaderSizeAsync(Stream stream, CancellationToken cancellationToken)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(sizeof(long));
        try
        {
            await stream.ReadExactlyAsync(buffer.AsMemory(0, sizeof(long)), cancellationToken).ConfigureAwait(false);
            var headerSize = BinaryPrimitives.ReadInt64LittleEndian(buffer.AsSpan(0, sizeof(long)));
            if (headerSize < 0)
            {
                throw new InvalidDataException($"Negative header size '{headerSize}' in safetensors file.");
            }

            return headerSize;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private static IReadOnlyList<SafeTensorMetadata> ParseTensorMetadata(JsonElement root)
    {
        var tensors = new List<SafeTensorMetadata>();
        foreach (var property in root.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal))
        {
            var dtype = property.Value.GetProperty("dtype").GetString()
                ?? throw new InvalidDataException($"Tensor '{property.Name}' is missing 'dtype'.");
            var offsets = property.Value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            if (offsets.Length != 2)
            {
                throw new InvalidDataException($"Tensor '{property.Name}' has invalid 'data_offsets'.");
            }

            var start = offsets[0];
            var end = offsets[1];
            if (end < start)
            {
                throw new InvalidDataException($"Tensor '{property.Name}' has decreasing data offsets.");
            }

            var shape = property.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            tensors.Add(new SafeTensorMetadata(property.Name, dtype, shape, start, end - start));
        }

        return tensors;
    }

    private static async Task<string> ComputeTensorSha256Async(
        Stream stream,
        long offset,
        long length,
        CancellationToken cancellationToken)
    {
        if (length < 0)
        {
            throw new InvalidDataException("Tensor length cannot be negative.");
        }

        var buffer = ArrayPool<byte>.Shared.Rent(128 * 1024);
        try
        {
            stream.Seek(offset, SeekOrigin.Begin);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var remaining = length;
            while (remaining > 0)
            {
                var toRead = (int)Math.Min(buffer.Length, remaining);
                var read = await stream.ReadAsync(buffer.AsMemory(0, toRead), cancellationToken).ConfigureAwait(false);
                if (read == 0)
                {
                    throw new EndOfStreamException("Unexpected end of file while reading safetensors tensor data.");
                }

                hash.AppendData(buffer.AsSpan(0, read));
                remaining -= read;
            }

            return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private static async Task<string> ComputeFileSha256Async(string path, CancellationToken cancellationToken)
    {
        await using var stream = File.OpenRead(path);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = ArrayPool<byte>.Shared.Rent(128 * 1024);
        try
        {
            while (true)
            {
                var read = await stream.ReadAsync(buffer.AsMemory(), cancellationToken).ConfigureAwait(false);
                if (read == 0)
                {
                    break;
                }

                hash.AppendData(buffer.AsSpan(0, read));
            }

            return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private sealed record SafeTensorMetadata(
        string Name,
        string Dtype,
        IReadOnlyList<long> Shape,
        long OffsetStart,
        long Length);

    private static string NormalizeDtype(string dtype)
    {
        return dtype switch
        {
            "F64" => "float64",
            "F32" => "float32",
            "F16" => "float16",
            "BF16" => "bfloat16",
            "I64" => "int64",
            "I32" => "int32",
            "I16" => "int16",
            "I8" => "int8",
            "U8" => "uint8",
            _ => dtype,
        };
    }
}
