using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace TableFormerTorchSharpSdk.PagePreparation;

public sealed class TableFormerPageInputPreparer
{
    public TableFormerPageInputSnapshot PreparePageInput(
        FileInfo imageFile,
        IReadOnlyList<TableFormerBoundingBox>? tableBoundingBoxes = null,
        IReadOnlyList<TableFormerPageToken>? tokens = null)
    {
        var decodedImage = TableFormerDecodedPageImage.Decode(imageFile);
        return PreparePageInput(decodedImage, tableBoundingBoxes, tokens);
    }

    public TableFormerPageInputSnapshot PreparePageInput(
        TableFormerDecodedPageImage decodedImage,
        IReadOnlyList<TableFormerBoundingBox>? tableBoundingBoxes = null,
        IReadOnlyList<TableFormerPageToken>? tokens = null)
    {
        ArgumentNullException.ThrowIfNull(decodedImage);

        var boundingBoxes = tableBoundingBoxes?.ToArray()
            ?? new[] { new TableFormerBoundingBox(0.0, 0.0, decodedImage.Width, decodedImage.Height) };

        var resolvedTokens = tokens?.ToArray() ?? Array.Empty<TableFormerPageToken>();

        return new TableFormerPageInputSnapshot(
            decodedImage.Width,
            decodedImage.Height,
            decodedImage.RgbBytes,
            decodedImage.ImageSha256,
            boundingBoxes,
            resolvedTokens);
    }
}
