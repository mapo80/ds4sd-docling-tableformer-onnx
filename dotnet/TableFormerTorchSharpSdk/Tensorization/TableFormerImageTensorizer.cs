using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

using TorchSharp;
using TorchScalarType = TorchSharp.torch.ScalarType;
using TorchTensor = TorchSharp.torch.Tensor;

using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.PagePreparation;

namespace TableFormerTorchSharpSdk.Tensorization;

public sealed class TableFormerImageTensorizer
{
    private const int Channels = 3;

    private readonly float[] _mean;
    private readonly float[] _std;
    private readonly int _targetSize;

    private TableFormerImageTensorizer(float[] mean, float[] std, int targetSize)
    {
        if (targetSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetSize), targetSize, "Target size must be positive.");
        }

        if (mean.Length != Channels || std.Length != Channels)
        {
            throw new ArgumentException($"Normalization arrays must contain {Channels} elements.");
        }

        _mean = mean.ToArray();
        _std = std.ToArray();
        _targetSize = targetSize;
    }

    public static TableFormerImageTensorizer FromConfig(TableFormerConfigSnapshot configSnapshot)
    {
        ArgumentNullException.ThrowIfNull(configSnapshot);

        if (configSnapshot.Config["dataset"] is not JsonObject datasetObject)
        {
            throw new InvalidDataException("Configuration is missing the 'dataset' section.");
        }

        if (datasetObject["image_normalization"] is not JsonObject normalizationObject)
        {
            throw new InvalidDataException("Configuration is missing the 'dataset.image_normalization' section.");
        }

        var mean = ExtractFloatArray(normalizationObject, "mean");
        var std = ExtractFloatArray(normalizationObject, "std");

        var targetSizeNode = datasetObject["resized_image"]
            ?? throw new InvalidDataException("Configuration is missing the 'dataset.resized_image' parameter.");

        var targetSize = Convert.ToInt32(targetSizeNode.GetValue<double>());
        return new TableFormerImageTensorizer(mean, std, targetSize);
    }

    public TableFormerImageTensorSnapshot CreateTensor(TableFormerTableCropSnapshot cropSnapshot)
    {
        ArgumentNullException.ThrowIfNull(cropSnapshot);

        var sourceWidth = cropSnapshot.RoundedPixelWidth;
        var sourceHeight = cropSnapshot.RoundedPixelHeight;
        if (sourceWidth <= 0 || sourceHeight <= 0)
        {
            throw new InvalidDataException(
                $"Crop snapshot dimensions must be positive but were {sourceWidth}x{sourceHeight}.");
        }

        ReadOnlySpan<float> cropValuesSpan;
        float[]? fallbackValues = null;
        if (cropSnapshot.CropFloatValues.Length == sourceWidth * sourceHeight * Channels)
        {
            cropValuesSpan = cropSnapshot.CropFloatValues;
        }
        else
        {
            var cropBytes = cropSnapshot.CropBytes;
            if (cropBytes.Length != sourceWidth * sourceHeight * Channels)
            {
                throw new InvalidDataException(
                    "Crop snapshot byte data does not match the expected RGB layout for the provided dimensions.");
            }

            fallbackValues = ConvertBytesToFloat(cropBytes);
            cropValuesSpan = fallbackValues;
        }

        var resizedPixels = ResizeCropValues(
            cropValuesSpan,
            sourceWidth,
            sourceHeight);

        var normalized = NormalizeResizedPixels(resizedPixels);
        var channelMajor = ConvertToChannelMajor(normalized);

        var sha256 = ComputeSha256(channelMajor);
        var tensor = TorchSharp.torch.tensor(channelMajor, new long[] { Channels, _targetSize, _targetSize }, dtype: TorchScalarType.Float32);
        var batched = tensor.unsqueeze(0);
        tensor.Dispose();

        return new TableFormerImageTensorSnapshot(
            batchSize: 1,
            channels: Channels,
            targetSize: _targetSize,
            tensorValues: channelMajor,
            tensor: batched,
            tensorSha256: sha256);
    }

    private float[] ResizeCropValues(ReadOnlySpan<float> sourceValues, int sourceWidth, int sourceHeight)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0)
        {
            throw new InvalidDataException(
                $"Crop dimensions must be positive but were {sourceWidth}x{sourceHeight}.");
        }

        if (sourceValues.Length != sourceWidth * sourceHeight * Channels)
        {
            throw new InvalidDataException(
                "Crop value length does not match the expected RGB layout for the provided dimensions.");
        }

        var destination = new float[_targetSize * _targetSize * Channels];

        if (sourceWidth == _targetSize && sourceHeight == _targetSize)
        {
            sourceValues.CopyTo(destination);
            return destination;
        }

        var scaleX = _targetSize / (double)sourceWidth;
        var scaleY = _targetSize / (double)sourceHeight;

        for (var y = 0; y < _targetSize; y++)
        {
            var srcY = y / scaleY;
            var y0 = (int)Math.Floor(srcY);
            double yLerp;

            if (y0 < 0)
            {
                y0 = 0;
                yLerp = 0.0;
            }
            else if (y0 >= sourceHeight - 1)
            {
                y0 = sourceHeight - 1;
                yLerp = 0.0;
            }
            else
            {
                yLerp = srcY - y0;
            }

            var y1 = Math.Min(y0 + 1, sourceHeight - 1);

            for (var x = 0; x < _targetSize; x++)
            {
                var srcX = x / scaleX;
                var x0 = (int)Math.Floor(srcX);
                double xLerp;

                if (x0 < 0)
                {
                    x0 = 0;
                    xLerp = 0.0;
                }
                else if (x0 >= sourceWidth - 1)
                {
                    x0 = sourceWidth - 1;
                    xLerp = 0.0;
                }
                else
                {
                    xLerp = srcX - x0;
                }

                var x1 = Math.Min(x0 + 1, sourceWidth - 1);

                var dstIndex = (y * _targetSize + x) * Channels;

                var idx00 = (y0 * sourceWidth + x0) * Channels;
                var idx10 = (y0 * sourceWidth + x1) * Channels;
                var idx01 = (y1 * sourceWidth + x0) * Channels;
                var idx11 = (y1 * sourceWidth + x1) * Channels;

                for (var channel = 0; channel < Channels; channel++)
                {
                    var p00 = sourceValues[idx00 + channel];
                    var p10 = sourceValues[idx10 + channel];
                    var p01 = sourceValues[idx01 + channel];
                    var p11 = sourceValues[idx11 + channel];

                    var top = p00 + (p10 - p00) * xLerp;
                    var bottom = p01 + (p11 - p01) * xLerp;
                    var value = top + (bottom - top) * yLerp;

                    destination[dstIndex + channel] = (float)value;
                }
            }
        }

        return destination;
    }

    private float[] NormalizeResizedPixels(float[] resizedPixels)
    {
        var normalized = new float[resizedPixels.Length];

        for (var i = 0; i < resizedPixels.Length; i++)
        {
            var channel = i % Channels;
            var clamped = Math.Clamp(resizedPixels[i], 0f, 255f);
            var value = clamped / 255f;
            normalized[i] = (value - _mean[channel]) / _std[channel];
        }

        return normalized;
    }

    private float[] ConvertToChannelMajor(float[] normalized)
    {
        var destination = new float[_targetSize * _targetSize * Channels];

        for (var channel = 0; channel < Channels; channel++)
        {
            for (var width = 0; width < _targetSize; width++)
            {
                for (var height = 0; height < _targetSize; height++)
                {
                    var sourceIndex = ((height * _targetSize) + width) * Channels + channel;
                    var destinationIndex = (channel * _targetSize + width) * _targetSize + height;
                    destination[destinationIndex] = normalized[sourceIndex];
                }
            }
        }

        return destination;
    }


    private static float[] ConvertBytesToFloat(byte[] source)
    {
        var values = new float[source.Length];
        for (var i = 0; i < source.Length; i++)
        {
            values[i] = source[i];
        }

        return values;
    }

    private static string ComputeSha256(float[] values)
    {
        var bytes = MemoryMarshal.AsBytes(values.AsSpan());
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static float[] ExtractFloatArray(JsonObject parent, string propertyName)
    {
        if (parent[propertyName] is not JsonArray array)
        {
            throw new InvalidDataException($"Configuration is missing the '{propertyName}' array.");
        }

        var result = new float[array.Count];
        for (var i = 0; i < array.Count; i++)
        {
            var node = array[i] ?? throw new InvalidDataException(
                $"Null entry encountered in configuration array '{propertyName}'.");

            result[i] = (float)node.GetValue<double>();
        }

        return result;
    }
}

public sealed class TableFormerImageTensorSnapshot : IDisposable
{
    private readonly float[] _tensorValues;

    public TableFormerImageTensorSnapshot(
        int batchSize,
        int channels,
        int targetSize,
        float[] tensorValues,
        TorchTensor tensor,
        string tensorSha256)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        }

        if (channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel count must be positive.");
        }

        if (targetSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetSize), targetSize, "Target size must be positive.");
        }

        ArgumentNullException.ThrowIfNull(tensorValues);
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentException.ThrowIfNullOrEmpty(tensorSha256);

        BatchSize = batchSize;
        Channels = channels;
        TargetSize = targetSize;
        _tensorValues = tensorValues;
        Tensor = tensor;
        TensorSha256 = tensorSha256;

        (MinValue, MaxValue, Mean, StandardDeviation) = ComputeStatistics(_tensorValues);
    }

    public int BatchSize { get; }

    public int Channels { get; }

    public int TargetSize { get; }

    public TorchTensor Tensor { get; }

    public string TensorSha256 { get; }

    public ReadOnlyMemory<float> TensorValues => _tensorValues;

    public int TensorLength => _tensorValues.Length;

    public float MinValue { get; }

    public float MaxValue { get; }

    public double Mean { get; }

    public double StandardDeviation { get; }

    public float[] GetTensorValuesCopy() => (float[])_tensorValues.Clone();

    public void Dispose()
    {
        Tensor.Dispose();
    }

    private static (float min, float max, double mean, double stdDev) ComputeStatistics(float[] values)
    {
        if (values.Length == 0)
        {
            return (0.0f, 0.0f, 0.0, 0.0);
        }

        var min = values[0];
        var max = values[0];
        double sum = 0.0;
        double sumSquares = 0.0;

        foreach (var value in values)
        {
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }

            sum += value;
            sumSquares += value * value;
        }

        var mean = sum / values.Length;
        var variance = Math.Max(0.0, (sumSquares / values.Length) - (mean * mean));
        var stdDev = Math.Sqrt(variance);
        return (min, max, mean, stdDev);
    }
}
