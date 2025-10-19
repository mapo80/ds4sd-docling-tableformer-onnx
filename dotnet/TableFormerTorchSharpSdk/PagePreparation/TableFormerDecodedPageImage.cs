using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

using SkiaSharp;

namespace TableFormerTorchSharpSdk.PagePreparation;

public sealed class TableFormerDecodedPageImage
{
    public TableFormerDecodedPageImage(int width, int height, byte[] rgbBytes, string imageSha256)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), width, "Width must be positive.");
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height), height, "Height must be positive.");
        }

        ArgumentNullException.ThrowIfNull(rgbBytes);
        ArgumentNullException.ThrowIfNull(imageSha256);

        if (string.IsNullOrWhiteSpace(imageSha256))
        {
            throw new ArgumentException("Image digest must be a non-empty SHA-256 string.", nameof(imageSha256));
        }

        var expectedLength = checked((long)width * height * 3);
        if (rgbBytes.LongLength != expectedLength)
        {
            throw new ArgumentException(
                $"RGB buffer length {rgbBytes.LongLength} does not match expected size {expectedLength}.",
                nameof(rgbBytes));
        }

        Width = width;
        Height = height;
        RgbBytes = rgbBytes.ToArray();
        ImageSha256 = imageSha256;
    }

    public int Width { get; }

    public int Height { get; }

    public byte[] RgbBytes { get; }

    public string ImageSha256 { get; }

    public static TableFormerDecodedPageImage Decode(FileInfo imageFile)
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

        return Decode(bitmap);
    }

    public static TableFormerDecodedPageImage Decode(SKBitmap bitmap)
    {
        ArgumentNullException.ThrowIfNull(bitmap);

        var resolvedBitmap = bitmap;
        var ownsResolvedBitmap = false;

        if (resolvedBitmap.ColorType != SKColorType.Rgba8888
            || (resolvedBitmap.AlphaType != SKAlphaType.Unpremul && resolvedBitmap.AlphaType != SKAlphaType.Opaque))
        {
            resolvedBitmap = new SKBitmap(new SKImageInfo(
                resolvedBitmap.Width,
                resolvedBitmap.Height,
                SKColorType.Rgba8888,
                SKAlphaType.Unpremul));

            if (!bitmap.CopyTo(resolvedBitmap, SKColorType.Rgba8888))
            {
                resolvedBitmap.Dispose();
                throw new InvalidDataException("Failed to convert bitmap to RGBA8888/Unpremul format.");
            }

            ownsResolvedBitmap = true;
        }

        try
        {
            using var pixmap = resolvedBitmap.PeekPixels();
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

            var width = pixmap.Width;
            var height = pixmap.Height;
            var bytesPerPixel = pixmap.BytesPerPixel;
            var rowBytes = pixmap.RowBytes;
            if (bytesPerPixel < 3)
            {
                throw new InvalidDataException(
                    $"Unexpected bytes-per-pixel value {bytesPerPixel}; an RGB(A) layout is required.");
            }

            var pixelBufferLength = checked(rowBytes * height);
            var pixelBytes = new byte[pixelBufferLength];
            var pixelPointer = pixmap.GetPixels();
            if (pixelPointer == IntPtr.Zero)
            {
                throw new InvalidDataException("Pixmap does not expose a valid pixel pointer.");
            }

            Marshal.Copy(pixelPointer, pixelBytes, 0, pixelBytes.Length);

            var rgbLength = checked(width * height * 3);
            var rgbBytes = new byte[rgbLength];
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
            return new TableFormerDecodedPageImage(width, height, rgbBytes, sha256);
        }
        finally
        {
            if (ownsResolvedBitmap)
            {
                resolvedBitmap.Dispose();
            }
        }
    }
}
