using System.Collections.ObjectModel;

namespace TableFormerTorchSharpSdk.PagePreparation;

public sealed class TableFormerPageInputSnapshot
{
    public TableFormerPageInputSnapshot(
        int width,
        int height,
        byte[] imageBytes,
        string imageSha256,
        IReadOnlyList<TableFormerBoundingBox> tableBoundingBoxes,
        IReadOnlyList<TableFormerPageToken> tokens)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), width, "Width must be positive.");
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height), height, "Height must be positive.");
        }

        ArgumentNullException.ThrowIfNull(imageBytes);
        ArgumentNullException.ThrowIfNull(imageSha256);
        ArgumentNullException.ThrowIfNull(tableBoundingBoxes);
        ArgumentNullException.ThrowIfNull(tokens);

        if (string.IsNullOrWhiteSpace(imageSha256))
        {
            throw new ArgumentException("Image digest must be a non-empty SHA-256 string.", nameof(imageSha256));
        }

        var expectedLength = checked((long)width * height * 3);
        if (imageBytes.LongLength != expectedLength)
        {
            throw new ArgumentException(
                $"Image buffer length {imageBytes.LongLength} does not match expected RGB size {expectedLength}.",
                nameof(imageBytes));
        }

        Width = width;
        Height = height;
        ImageBytes = imageBytes.ToArray();
        ImageSha256 = imageSha256;
        TableBoundingBoxes = new ReadOnlyCollection<TableFormerBoundingBox>(tableBoundingBoxes.ToArray());
        Tokens = new ReadOnlyCollection<TableFormerPageToken>(tokens.ToArray());
    }

    public int Width { get; }

    public int Height { get; }

    public byte[] ImageBytes { get; }

    public string ImageSha256 { get; }

    public IReadOnlyList<TableFormerBoundingBox> TableBoundingBoxes { get; }

    public IReadOnlyList<TableFormerPageToken> Tokens { get; }
}

public sealed record TableFormerBoundingBox(double Left, double Top, double Right, double Bottom)
{
    public double[] ToArray() => new[] { Left, Top, Right, Bottom };
}

public sealed record TableFormerPageToken();
