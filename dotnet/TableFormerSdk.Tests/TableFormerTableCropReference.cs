using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace TableFormerSdk.Tests;

internal sealed record TableFormerTableCropReference(IReadOnlyList<TableFormerTableCropSample> Samples)
{
    public static TableFormerTableCropReference Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        if (!document.RootElement.TryGetProperty("samples", out var samplesElement))
        {
            throw new InvalidDataException("Reference JSON does not contain a 'samples' array.");
        }

        var samples = new List<TableFormerTableCropSample>();
        foreach (var sampleElement in samplesElement.EnumerateArray())
        {
            samples.Add(ParseSample(sampleElement));
        }

        return new TableFormerTableCropReference(samples);
    }

    private static TableFormerTableCropSample ParseSample(JsonElement element)
    {
        var imageName = element.GetProperty("image_name").GetString()
            ?? throw new InvalidDataException("Sample is missing 'image_name'.");
        var originalWidth = element.GetProperty("original_width").GetDouble();
        var originalHeight = element.GetProperty("original_height").GetDouble();
        var scaleFactor = element.GetProperty("scale_factor").GetDouble();
        var resizedWidth = element.GetProperty("resized_width").GetInt32();
        var resizedHeight = element.GetProperty("resized_height").GetInt32();

        var tableCropsElement = element.GetProperty("table_crops");
        var tableCrops = new List<TableFormerTableCropEntry>(tableCropsElement.GetArrayLength());
        foreach (var cropElement in tableCropsElement.EnumerateArray())
        {
            tableCrops.Add(ParseCrop(imageName, cropElement));
        }

        return new TableFormerTableCropSample(
            imageName,
            originalWidth,
            originalHeight,
            scaleFactor,
            resizedWidth,
            resizedHeight,
            tableCrops);
    }

    private static TableFormerTableCropEntry ParseCrop(string imageName, JsonElement element)
    {
        var tableIndex = element.GetProperty("table_index").GetInt32();

        static double[] ParseDoubleArray(JsonElement arrayElement, string description)
        {
            var values = new double[arrayElement.GetArrayLength()];
            var i = 0;
            foreach (var value in arrayElement.EnumerateArray())
            {
                values[i++] = value.GetDouble();
            }

            return values;
        }

        static int[] ParseIntArray(JsonElement arrayElement, string description)
        {
            var values = new int[arrayElement.GetArrayLength()];
            var i = 0;
            foreach (var value in arrayElement.EnumerateArray())
            {
                values[i++] = value.GetInt32();
            }

            return values;
        }

        var originalBBox = ParseDoubleArray(element.GetProperty("original_bbox"), "original_bbox");
        var scaledBBox = ParseDoubleArray(element.GetProperty("scaled_bbox"), "scaled_bbox");
        var roundedBBox = ParseIntArray(element.GetProperty("rounded_bbox"), "rounded_bbox");

        if (originalBBox.Length != 4 || scaledBBox.Length != 4 || roundedBBox.Length != 4)
        {
            throw new InvalidDataException(
                $"Crop entry for '{imageName}' table index {tableIndex} contains malformed bounding boxes.");
        }

        var cropWidth = element.GetProperty("crop_width").GetInt32();
        var cropHeight = element.GetProperty("crop_height").GetInt32();
        var cropByteLength = element.GetProperty("crop_byte_length").GetInt32();
        var cropSha256 = element.GetProperty("crop_image_sha256").GetString()
            ?? throw new InvalidDataException(
                $"Crop entry for '{imageName}' table index {tableIndex} is missing 'crop_image_sha256'.");

        var cropMean = element.GetProperty("crop_mean_pixel_value").GetDouble();
        var cropStd = element.GetProperty("crop_std_pixel_value").GetDouble();
        var cropChannels = element.GetProperty("crop_channels").GetInt32();

        return new TableFormerTableCropEntry(
            tableIndex,
            originalBBox,
            scaledBBox,
            roundedBBox,
            element.GetProperty("original_bbox_pixel_width").GetDouble(),
            element.GetProperty("original_bbox_pixel_height").GetDouble(),
            element.GetProperty("scaled_bbox_pixel_width").GetDouble(),
            element.GetProperty("scaled_bbox_pixel_height").GetDouble(),
            element.GetProperty("rounded_bbox_pixel_width").GetInt32(),
            element.GetProperty("rounded_bbox_pixel_height").GetInt32(),
            cropWidth,
            cropHeight,
            cropByteLength,
            cropSha256,
            cropMean,
            cropStd,
            cropChannels);
    }
}

internal sealed record TableFormerTableCropSample(
    string ImageName,
    double OriginalWidth,
    double OriginalHeight,
    double ScaleFactor,
    int ResizedWidth,
    int ResizedHeight,
    IReadOnlyList<TableFormerTableCropEntry> TableCrops);

internal sealed record TableFormerTableCropEntry(
    int TableIndex,
    IReadOnlyList<double> OriginalBoundingBox,
    IReadOnlyList<double> ScaledBoundingBox,
    IReadOnlyList<int> RoundedBoundingBox,
    double OriginalPixelWidth,
    double OriginalPixelHeight,
    double ScaledPixelWidth,
    double ScaledPixelHeight,
    int RoundedPixelWidth,
    int RoundedPixelHeight,
    int CropWidth,
    int CropHeight,
    int CropByteLength,
    string CropSha256,
    double CropMean,
    double CropStd,
    int CropChannels);
