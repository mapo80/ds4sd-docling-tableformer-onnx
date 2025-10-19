using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace TableFormerSdk.Tests;

internal sealed record TableFormerPageInputReference(IReadOnlyList<TableFormerPageInputSample> Samples)
{
    public static TableFormerPageInputReference Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        if (!document.RootElement.TryGetProperty("samples", out var samplesElement))
        {
            throw new InvalidDataException("Reference JSON does not contain a 'samples' array.");
        }

        var samples = new List<TableFormerPageInputSample>();
        foreach (var sampleElement in samplesElement.EnumerateArray())
        {
            samples.Add(ParseSample(sampleElement));
        }

        return new TableFormerPageInputReference(samples);
    }

    private static TableFormerPageInputSample ParseSample(JsonElement element)
    {
        var imageName = element.GetProperty("image_name").GetString()
            ?? throw new InvalidDataException("Sample is missing 'image_name'.");
        var width = element.GetProperty("width").GetDouble();
        var height = element.GetProperty("height").GetDouble();
        var imageBytes = Convert.FromBase64String(
            element.GetProperty("image_bytes_base64").GetString()
            ?? throw new InvalidDataException($"Sample '{imageName}' is missing base64 image data."));
        var imageSha256 = element.GetProperty("image_sha256").GetString()
            ?? throw new InvalidDataException($"Sample '{imageName}' is missing 'image_sha256'.");

        var tableBboxesElement = element.GetProperty("table_bboxes");
        var tableBoundingBoxes = new List<double[]>(tableBboxesElement.GetArrayLength());
        foreach (var bboxElement in tableBboxesElement.EnumerateArray())
        {
            var bbox = new double[4];
            var index = 0;
            foreach (var value in bboxElement.EnumerateArray())
            {
                if (index >= 4)
                {
                    throw new InvalidDataException(
                        $"Bounding box for '{imageName}' contains more than four elements.");
                }

                bbox[index++] = value.GetDouble();
            }

            if (index != 4)
            {
                throw new InvalidDataException(
                    $"Bounding box for '{imageName}' does not contain four elements.");
            }

            tableBoundingBoxes.Add(bbox);
        }

        var tokensCount = element.GetProperty("tokens").GetArrayLength();

        var byteLength = element.GetProperty("byte_length").GetInt64();
        if (byteLength != imageBytes.LongLength)
        {
            throw new InvalidDataException(
                $"Sample '{imageName}' reports byte_length {byteLength} but decoded length is {imageBytes.LongLength}.");
        }

        return new TableFormerPageInputSample(
            imageName,
            width,
            height,
            tableBoundingBoxes,
            imageBytes,
            imageSha256,
            tokensCount);
    }
}

internal sealed record TableFormerPageInputSample(
    string ImageName,
    double Width,
    double Height,
    IReadOnlyList<double[]> TableBoundingBoxes,
    byte[] ImageBytes,
    string ImageSha256,
    int TokenCount);
