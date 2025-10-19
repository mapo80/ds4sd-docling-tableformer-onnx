using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

using SkiaSharp;

namespace TableFormerTorchSharpSdk.PagePreparation;

public sealed class TableFormerPageInputPreparer
{
    public TableFormerPageInputSnapshot PreparePageInput(
        FileInfo imageFile,
        IReadOnlyList<TableFormerBoundingBox>? tableBoundingBoxes = null,
        IReadOnlyList<TableFormerPageToken>? tokens = null)
    {
        ArgumentNullException.ThrowIfNull(imageFile);
        if (!imageFile.Exists)
        {
            throw new FileNotFoundException($"Image file not found: '{imageFile.FullName}'.", imageFile.FullName);
        }

        using var stream = imageFile.OpenRead();
        using var bitmap = SKBitmap.Decode(stream);
        if (bitmap is null)
        {
            throw new InvalidDataException($"Unable to decode image '{imageFile.FullName}'.");
        }

        if (bitmap.ColorType != SKColorType.Rgba8888)
        {
            using var converted = new SKBitmap(new SKImageInfo(bitmap.Width, bitmap.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul));
            if (!bitmap.CopyTo(converted, SKColorType.Rgba8888))
            {
                throw new InvalidDataException(
                    $"Failed to convert image '{imageFile.FullName}' to RGBA8888/Unpremul format.");
            }

            return PrepareFromBitmap(converted, tableBoundingBoxes, tokens);
        }

        return PrepareFromBitmap(bitmap, tableBoundingBoxes, tokens);
    }

    private TableFormerPageInputSnapshot PrepareFromBitmap(
        SKBitmap bitmap,
        IReadOnlyList<TableFormerBoundingBox>? tableBoundingBoxes,
        IReadOnlyList<TableFormerPageToken>? tokens)
    {
        var width = bitmap.Width;
        var height = bitmap.Height;
        using var pixmap = bitmap.PeekPixels();
        if (pixmap is null)
        {
            throw new InvalidDataException("Unable to access bitmap pixel data.");
        }

        if (pixmap.ColorType != SKColorType.Rgba8888
            || (pixmap.AlphaType != SKAlphaType.Unpremul && pixmap.AlphaType != SKAlphaType.Opaque))
        {
            throw new InvalidDataException(
                $"Pixmap color space mismatch. Expected RGBA8888 with Unpremul or Opaque alpha but found {pixmap.ColorType}/{pixmap.AlphaType}.");
        }

        var bufferLength = checked((long)width * height * 3);
        if (bufferLength > int.MaxValue)
        {
            throw new InvalidDataException(
                $"Image '{bitmap.Width}x{bitmap.Height}' is too large to fit into a contiguous RGB buffer.");
        }

        var rgbBytes = new byte[(int)bufferLength];
        var rowBytes = pixmap.RowBytes;
        var bytesPerPixel = pixmap.BytesPerPixel;
        if (bytesPerPixel < 3)
        {
            throw new InvalidDataException(
                $"Unexpected bytes-per-pixel value {bytesPerPixel}; an RGB(A) layout is required.");
        }

        var pixelBufferLength = checked((long)rowBytes * height);
        if (pixelBufferLength > int.MaxValue)
        {
            throw new InvalidDataException(
                $"Pixel buffer length {pixelBufferLength} exceeds supported range.");
        }

        var pixelBytes = new byte[(int)pixelBufferLength];
        var pixelPointer = pixmap.GetPixels();
        if (pixelPointer == IntPtr.Zero)
        {
            throw new InvalidDataException("Pixmap does not expose a valid pixel pointer.");
        }

        Marshal.Copy(pixelPointer, pixelBytes, 0, pixelBytes.Length);

        var destinationIndex = 0;
        for (var y = 0; y < height; y++)
        {
            var rowOffset = y * rowBytes;
            for (var x = 0; x < width; x++)
            {
                var offset = rowOffset + x * bytesPerPixel;
                rgbBytes[destinationIndex++] = pixelBytes[offset + 0];
                rgbBytes[destinationIndex++] = pixelBytes[offset + 1];
                rgbBytes[destinationIndex++] = pixelBytes[offset + 2];
            }
        }

        var sha256 = Convert.ToHexString(SHA256.HashData(rgbBytes)).ToLowerInvariant();

        var boundingBoxes = tableBoundingBoxes?.ToArray()
            ?? new[] { new TableFormerBoundingBox(0.0, 0.0, width, height) };

        var resolvedTokens = tokens?.ToArray() ?? Array.Empty<TableFormerPageToken>();

        return new TableFormerPageInputSnapshot(
            width,
            height,
            rgbBytes,
            sha256,
            boundingBoxes,
            resolvedTokens);
    }
}
