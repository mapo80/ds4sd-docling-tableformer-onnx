using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace TableFormerTorchSharpSdk.PagePreparation;

public sealed class TableFormerTableCropper
{
    private const int TargetHeight = 1024;

    public TableFormerPageResizeSnapshot PrepareTableCrops(
        FileInfo imageFile,
        IReadOnlyList<TableFormerBoundingBox>? tableBoundingBoxes = null)
    {
        var decodedImage = TableFormerDecodedPageImage.Decode(imageFile);
        return PrepareTableCrops(decodedImage, tableBoundingBoxes);
    }

    public TableFormerPageResizeSnapshot PrepareTableCrops(
        TableFormerDecodedPageImage decodedImage,
        IReadOnlyList<TableFormerBoundingBox>? tableBoundingBoxes = null)
    {
        ArgumentNullException.ThrowIfNull(decodedImage);

        var originalWidth = decodedImage.Width;
        var originalHeight = decodedImage.Height;

        var scaleFactor = TargetHeight / (double)originalHeight;
        var resizedWidth = (int)(originalWidth * scaleFactor);
        if (resizedWidth <= 0)
        {
            throw new InvalidDataException(
                $"Computed resized width {resizedWidth} is invalid for image {originalWidth}x{originalHeight}.");
        }

        var resizedRgb = ResizeRgbBuffer(
            decodedImage.RgbBytes,
            originalWidth,
            originalHeight,
            resizedWidth,
            TargetHeight,
            out var resizedFloatRgb);

        var resolvedBoundingBoxes = tableBoundingBoxes?.ToArray()
            ?? new[] { new TableFormerBoundingBox(0.0, 0.0, originalWidth, originalHeight) };

        var cropSnapshots = new List<TableFormerTableCropSnapshot>(resolvedBoundingBoxes.Length);
        foreach (var bbox in resolvedBoundingBoxes)
        {
            cropSnapshots.Add(CreateCropSnapshot(
                resizedRgb,
                resizedFloatRgb,
                resizedWidth,
                TargetHeight,
                scaleFactor,
                bbox));
        }

        return new TableFormerPageResizeSnapshot(
            originalWidth,
            originalHeight,
            resizedWidth,
            TargetHeight,
            scaleFactor,
            cropSnapshots);
    }

    private static TableFormerTableCropSnapshot CreateCropSnapshot(
        byte[] resizedRgb,
        float[] resizedFloatRgb,
        int resizedWidth,
        int resizedHeight,
        double scaleFactor,
        TableFormerBoundingBox originalBoundingBox)
    {
        var scaledBoundingBox = new TableFormerBoundingBox(
            originalBoundingBox.Left * scaleFactor,
            originalBoundingBox.Top * scaleFactor,
            originalBoundingBox.Right * scaleFactor,
            originalBoundingBox.Bottom * scaleFactor);

        var roundedBoundingBox = TableFormerRoundedBoundingBox.FromScaledBoundingBox(
            scaledBoundingBox,
            resizedWidth,
            resizedHeight);

        var cropBytes = ExtractCropBytes(resizedRgb, resizedWidth, resizedHeight, roundedBoundingBox);
        var cropFloats = ExtractCropFloats(resizedFloatRgb, resizedWidth, resizedHeight, roundedBoundingBox);
        var cropSha256 = Convert.ToHexString(SHA256.HashData(cropBytes)).ToLowerInvariant();

        var originalWidth = originalBoundingBox.Right - originalBoundingBox.Left;
        var originalHeight = originalBoundingBox.Bottom - originalBoundingBox.Top;
        var scaledWidth = scaledBoundingBox.Right - scaledBoundingBox.Left;
        var scaledHeight = scaledBoundingBox.Bottom - scaledBoundingBox.Top;
        var roundedWidth = roundedBoundingBox.Right - roundedBoundingBox.Left;
        var roundedHeight = roundedBoundingBox.Bottom - roundedBoundingBox.Top;

        return new TableFormerTableCropSnapshot(
            originalBoundingBox,
            scaledBoundingBox,
            roundedBoundingBox,
            originalWidth,
            originalHeight,
            scaledWidth,
            scaledHeight,
            roundedWidth,
            roundedHeight,
            cropBytes,
            cropFloats,
            cropSha256);
    }

    private static byte[] ResizeRgbBuffer(
        byte[] sourceRgb,
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        out float[] floatDestination)
    {
        const int channels = 3;
        var length = targetWidth * targetHeight * channels;
        var destination = new byte[length];
        floatDestination = new float[length];
        var floatBuffer = floatDestination;
        var byteBuffer = destination;

        if (sourceWidth == targetWidth && sourceHeight == targetHeight)
        {
            var copyLength = Math.Min(length, sourceRgb.Length);
            for (var i = 0; i < copyLength; i++)
            {
                var value = sourceRgb[i];
                byteBuffer[i] = value;
                floatBuffer[i] = value;
            }

            return byteBuffer;
        }

        var xWeights = BuildBilinearWeights(targetWidth, sourceWidth);
        var yWeights = BuildBilinearWeights(targetHeight, sourceHeight);

        Parallel.For(0, targetHeight, y =>
        {
            var yWeight = yWeights[y];
            var row0Base = yWeight.Index0 * sourceWidth * channels;
            var row1Base = yWeight.Index1 * sourceWidth * channels;
            var yLerp = yWeight.Weight;
            var destinationRowOffset = y * targetWidth * channels;

            for (var x = 0; x < targetWidth; x++)
            {
                var xWeight = xWeights[x];
                var x0Offset = xWeight.Index0 * channels;
                var x1Offset = xWeight.Index1 * channels;
                var destinationIndex = destinationRowOffset + x * channels;
                var xLerp = xWeight.Weight;

                for (var channel = 0; channel < channels; channel++)
                {
                    var p00 = sourceRgb[row0Base + x0Offset + channel];
                    var p10 = sourceRgb[row0Base + x1Offset + channel];
                    var p01 = sourceRgb[row1Base + x0Offset + channel];
                    var p11 = sourceRgb[row1Base + x1Offset + channel];

                    var top = p00 + (p10 - p00) * xLerp;
                    var bottom = p01 + (p11 - p01) * xLerp;
                    var value = top + (bottom - top) * yLerp;
                    var clamped = Math.Clamp(value, 0.0, 255.0);

                    floatBuffer[destinationIndex + channel] = (float)clamped;
                    var rounded = Math.Clamp((int)Math.Round(clamped), 0, 255);
                    byteBuffer[destinationIndex + channel] = (byte)rounded;
                }
            }
        });

        return byteBuffer;
    }

    private static BilinearWeight[] BuildBilinearWeights(int targetLength, int sourceLength)
    {
        var weights = new BilinearWeight[targetLength];
        if (targetLength == 0)
        {
            return weights;
        }

        var scale = targetLength / (double)sourceLength;
        var maxIndex = Math.Max(0, sourceLength - 1);

        for (var i = 0; i < targetLength; i++)
        {
            var src = i / scale;
            var index0 = (int)Math.Floor(src);
            var weight = src - index0;

            if (index0 < 0)
            {
                index0 = 0;
                weight = 0d;
            }
            else if (index0 >= maxIndex)
            {
                index0 = maxIndex;
                weight = 0d;
            }

            var index1 = Math.Min(index0 + 1, maxIndex);
            weights[i] = new BilinearWeight(index0, index1, weight);
        }

        return weights;
    }

    private readonly struct BilinearWeight
    {
        public BilinearWeight(int index0, int index1, double weight)
        {
            Index0 = index0;
            Index1 = index1;
            Weight = weight;
        }

        public int Index0 { get; }

        public int Index1 { get; }

        public double Weight { get; }
    }

    private static byte[] ExtractCropBytes(
        byte[] resizedRgb,
        int resizedWidth,
        int resizedHeight,
        TableFormerRoundedBoundingBox bbox)
    {
        var cropWidth = bbox.Right - bbox.Left;
        var cropHeight = bbox.Bottom - bbox.Top;
        if (cropWidth <= 0 || cropHeight <= 0)
        {
            throw new InvalidDataException(
                $"Crop dimensions must be positive but were {cropWidth}x{cropHeight}.");
        }

        if (bbox.Right > resizedWidth || bbox.Bottom > resizedHeight)
        {
            throw new InvalidDataException(
                $"Crop bounding box {bbox.Left},{bbox.Top},{bbox.Right},{bbox.Bottom} exceeds resized dimensions {resizedWidth}x{resizedHeight}.");
        }

        var channels = 3;
        var destination = new byte[cropWidth * cropHeight * channels];
        for (var row = 0; row < cropHeight; row++)
        {
            var sourceOffset = ((bbox.Top + row) * resizedWidth + bbox.Left) * channels;
            var destinationOffset = row * cropWidth * channels;
            Buffer.BlockCopy(resizedRgb, sourceOffset, destination, destinationOffset, cropWidth * channels);
        }

        return destination;
    }


    private static float[] ExtractCropFloats(
        float[] resizedFloats,
        int resizedWidth,
        int resizedHeight,
        TableFormerRoundedBoundingBox bbox)
    {
        var cropWidth = bbox.Right - bbox.Left;
        var cropHeight = bbox.Bottom - bbox.Top;
        if (cropWidth <= 0 || cropHeight <= 0)
        {
            throw new InvalidDataException(
                $"Crop dimensions must be positive but were {cropWidth}x{cropHeight}.");
        }

        if (bbox.Right > resizedWidth || bbox.Bottom > resizedHeight)
        {
            throw new InvalidDataException(
                $"Crop bounding box {bbox.Left},{bbox.Top},{bbox.Right},{bbox.Bottom} exceeds resized dimensions {resizedWidth}x{resizedHeight}.");
        }

        var channels = 3;
        var destination = new float[cropWidth * cropHeight * channels];
        for (var row = 0; row < cropHeight; row++)
        {
            var sourceOffset = ((bbox.Top + row) * resizedWidth + bbox.Left) * channels;
            var destinationOffset = row * cropWidth * channels;
            Array.Copy(resizedFloats, sourceOffset, destination, destinationOffset, cropWidth * channels);
        }

        return destination;
    }
}

public sealed class TableFormerPageResizeSnapshot
{
    public TableFormerPageResizeSnapshot(
        int originalWidth,
        int originalHeight,
        int resizedWidth,
        int resizedHeight,
        double scaleFactor,
        IReadOnlyList<TableFormerTableCropSnapshot> tableCrops)
    {
        if (originalWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalWidth), originalWidth, "Width must be positive.");
        }

        if (originalHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalHeight), originalHeight, "Height must be positive.");
        }

        if (resizedWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(resizedWidth), resizedWidth, "Resized width must be positive.");
        }

        if (resizedHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(resizedHeight), resizedHeight, "Resized height must be positive.");
        }

        if (scaleFactor <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), scaleFactor, "Scale factor must be positive.");
        }

        ArgumentNullException.ThrowIfNull(tableCrops);

        OriginalWidth = originalWidth;
        OriginalHeight = originalHeight;
        ResizedWidth = resizedWidth;
        ResizedHeight = resizedHeight;
        ScaleFactor = scaleFactor;
        TableCrops = new ReadOnlyCollection<TableFormerTableCropSnapshot>(tableCrops.ToArray());
    }

    public int OriginalWidth { get; }

    public int OriginalHeight { get; }

    public int ResizedWidth { get; }

    public int ResizedHeight { get; }

    public double ScaleFactor { get; }

    public IReadOnlyList<TableFormerTableCropSnapshot> TableCrops { get; }
}

public sealed class TableFormerTableCropSnapshot
{
    public TableFormerTableCropSnapshot(
        TableFormerBoundingBox originalBoundingBox,
        TableFormerBoundingBox scaledBoundingBox,
        TableFormerRoundedBoundingBox roundedBoundingBox,
        double originalPixelWidth,
        double originalPixelHeight,
        double scaledPixelWidth,
        double scaledPixelHeight,
        int roundedPixelWidth,
        int roundedPixelHeight,
        byte[] cropBytes,
        float[] cropFloatValues,
        string cropSha256)
    {
        ArgumentNullException.ThrowIfNull(cropBytes);
        ArgumentNullException.ThrowIfNull(cropFloatValues);
        ArgumentNullException.ThrowIfNull(cropSha256);

        OriginalBoundingBox = originalBoundingBox;
        ScaledBoundingBox = scaledBoundingBox;
        RoundedBoundingBox = roundedBoundingBox;
        OriginalPixelWidth = originalPixelWidth;
        OriginalPixelHeight = originalPixelHeight;
        ScaledPixelWidth = scaledPixelWidth;
        ScaledPixelHeight = scaledPixelHeight;
        RoundedPixelWidth = roundedPixelWidth;
        RoundedPixelHeight = roundedPixelHeight;
        CropBytes = cropBytes;
        CropFloatValues = cropFloatValues;
        CropSha256 = cropSha256;
    }

    public TableFormerBoundingBox OriginalBoundingBox { get; }

    public TableFormerBoundingBox ScaledBoundingBox { get; }

    public TableFormerRoundedBoundingBox RoundedBoundingBox { get; }

    public double OriginalPixelWidth { get; }

    public double OriginalPixelHeight { get; }

    public double ScaledPixelWidth { get; }

    public double ScaledPixelHeight { get; }

    public int RoundedPixelWidth { get; }

    public int RoundedPixelHeight { get; }

    public byte[] CropBytes { get; }

    public float[] CropFloatValues { get; }

    public string CropSha256 { get; }

    public int CropByteLength => CropBytes.Length;

    public int CropFloatLength => CropFloatValues.Length;
}

public sealed record TableFormerRoundedBoundingBox(int Left, int Top, int Right, int Bottom)
{
    public static TableFormerRoundedBoundingBox FromScaledBoundingBox(
        TableFormerBoundingBox bbox,
        int maxWidth,
        int maxHeight)
    {
        var left = ClampToBounds((int)Math.Round(bbox.Left, MidpointRounding.ToEven), 0, maxWidth);
        var top = ClampToBounds((int)Math.Round(bbox.Top, MidpointRounding.ToEven), 0, maxHeight);
        var right = ClampToBounds((int)Math.Round(bbox.Right, MidpointRounding.ToEven), 0, maxWidth);
        var bottom = ClampToBounds((int)Math.Round(bbox.Bottom, MidpointRounding.ToEven), 0, maxHeight);

        right = Math.Max(right, left);
        bottom = Math.Max(bottom, top);

        return new TableFormerRoundedBoundingBox(left, top, right, bottom);
    }

    private static int ClampToBounds(int value, int min, int max)
    {
        if (value < min)
        {
            return min;
        }

        if (value > max)
        {
            return max;
        }

        return value;
    }

    public int[] ToArray() => new[] { Left, Top, Right, Bottom };
}
