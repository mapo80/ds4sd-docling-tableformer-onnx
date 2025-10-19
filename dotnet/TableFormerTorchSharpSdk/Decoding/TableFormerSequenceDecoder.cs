using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;

using TableFormerTorchSharpSdk.Initialization;
using TableFormerTorchSharpSdk.Model;

using TorchSharp;
using torch = TorchSharp.torch;

namespace TableFormerTorchSharpSdk.Decoding;

internal sealed class TableFormerSequenceDecoder
{
    private readonly IReadOnlyDictionary<int, string> _reverseWordMapTag;

    public TableFormerSequenceDecoder(TableFormerInitializationSnapshot initializationSnapshot)
    {
        ArgumentNullException.ThrowIfNull(initializationSnapshot);
        _reverseWordMapTag = initializationSnapshot.WordMap.ReverseTagMap;
    }

    public TableFormerSequencePrediction Decode(TableFormerNeuralPrediction prediction)
    {
        ArgumentNullException.ThrowIfNull(prediction);

        var tagSequence = prediction.Sequence.ToArray();
        var rsSequence = MapToTags(tagSequence);
        var htmlSequence = OtslHtmlConverter.ToHtml(rsSequence);
        var rawBoundingBoxes = ConvertBoundingBoxes(prediction.Coordinates);
        var (synced, finalBoundingBoxes) = EnsureBoundingBoxSync(htmlSequence, rawBoundingBoxes);

        return new TableFormerSequencePrediction(
            tagSequence,
            rsSequence,
            htmlSequence,
            rawBoundingBoxes,
            finalBoundingBoxes,
            synced);
    }

    private IReadOnlyList<string> MapToTags(IReadOnlyList<int> sequence)
    {
        if (sequence.Count <= 2)
        {
            return Array.Empty<string>();
        }

        var tags = new string[sequence.Count - 2];
        for (var i = 1; i < sequence.Count - 1; i++)
        {
            var index = sequence[i];
            if (!_reverseWordMapTag.TryGetValue(index, out var tag))
            {
                throw new InvalidDataException($"Word map is missing tag index {index}.");
            }

            tags[i - 1] = tag;
        }

        return new ReadOnlyCollection<string>(tags);
    }

    private static IReadOnlyList<TableFormerNormalizedBoundingBox> ConvertBoundingBoxes(torch.Tensor coordinates)
    {
        if (coordinates.numel() == 0)
        {
            return Array.Empty<TableFormerNormalizedBoundingBox>();
        }

        using var coordsFloat = coordinates.to_type(torch.ScalarType.Float32);
        var components = coordsFloat.unbind(-1);
        using var xCenter = components[0];
        using var yCenter = components[1];
        using var width = components[2];
        using var height = components[3];

        using var halfWidth = width.mul(0.5f);
        using var halfHeight = height.mul(0.5f);

        using var leftTensor = xCenter - halfWidth;
        using var topTensor = yCenter - halfHeight;
        using var rightTensor = xCenter + halfWidth;
        using var bottomTensor = yCenter + halfHeight;

        using var stacked = torch.stack(new[] { leftTensor, topTensor, rightTensor, bottomTensor }, dim: -1);
        using var stackedCpu = stacked.to(torch.CPU);
        var values = stackedCpu.data<float>().ToArray();

        if (values.Length % 4 != 0)
        {
            throw new InvalidDataException(
                $"Bounding box tensor contains {values.Length} elements which is not divisible by 4.");
        }

        var boxes = new TableFormerNormalizedBoundingBox[values.Length / 4];
        for (var i = 0; i < boxes.Length; i++)
        {
            var left = values[i * 4 + 0];
            var top = values[i * 4 + 1];
            var right = values[i * 4 + 2];
            var bottom = values[i * 4 + 3];

            boxes[i] = new TableFormerNormalizedBoundingBox(left, top, right, bottom);
        }

        foreach (var component in components)
        {
            component.Dispose();
        }

        return new ReadOnlyCollection<TableFormerNormalizedBoundingBox>(boxes);
    }

    private static (bool Synced, IReadOnlyList<TableFormerNormalizedBoundingBox> Boxes) EnsureBoundingBoxSync(
        IReadOnlyList<string> htmlSequence,
        IReadOnlyList<TableFormerNormalizedBoundingBox> rawBoxes)
    {
        var expectedCount = CountCells(htmlSequence);
        if (expectedCount == rawBoxes.Count)
        {
            return (true, rawBoxes);
        }

        var corrected = RemoveSpanDesync(htmlSequence, rawBoxes);
        return (false, corrected);
    }

    private static int CountCells(IReadOnlyList<string> htmlSequence)
    {
        var count = 0;
        foreach (var token in htmlSequence)
        {
            if (token == "<td>" || token == ">")
            {
                count += 1;
            }

            if (token == "fcel" || token == "ecel" || token == "ched" || token == "rhed" || token == "srow")
            {
                count += 1;
            }
        }

        return count;
    }

    private static IReadOnlyList<TableFormerNormalizedBoundingBox> RemoveSpanDesync(
        IReadOnlyList<string> htmlSequence,
        IReadOnlyList<TableFormerNormalizedBoundingBox> rawBoxes)
    {
        var indexesToDelete = new HashSet<int>();
        var indexToDeleteFrom = 0;

        foreach (var token in htmlSequence)
        {
            if (token == "<td>")
            {
                indexToDeleteFrom += 1;
            }

            if (token == ">")
            {
                indexToDeleteFrom += 1;
                indexesToDelete.Add(indexToDeleteFrom);
            }
        }

        var corrected = new List<TableFormerNormalizedBoundingBox>(rawBoxes.Count);
        for (var i = 0; i < rawBoxes.Count; i++)
        {
            if (!indexesToDelete.Contains(i))
            {
                corrected.Add(rawBoxes[i]);
            }
        }

        return corrected.AsReadOnly();
    }
}

internal sealed record TableFormerNormalizedBoundingBox(double Left, double Top, double Right, double Bottom)
{
    public double[] ToArray() => new[] { Left, Top, Right, Bottom };
}

internal sealed class TableFormerSequencePrediction
{
    public TableFormerSequencePrediction(
        IReadOnlyList<int> tagSequence,
        IReadOnlyList<string> rsSequence,
        IReadOnlyList<string> htmlSequence,
        IReadOnlyList<TableFormerNormalizedBoundingBox> rawBoundingBoxes,
        IReadOnlyList<TableFormerNormalizedBoundingBox> finalBoundingBoxes,
        bool boundingBoxesSynced)
    {
        TagSequence = tagSequence;
        RsSequence = rsSequence;
        HtmlSequence = htmlSequence;
        RawBoundingBoxes = rawBoundingBoxes;
        FinalBoundingBoxes = finalBoundingBoxes;
        BoundingBoxesSynced = boundingBoxesSynced;
    }

    public IReadOnlyList<int> TagSequence { get; }

    public IReadOnlyList<string> RsSequence { get; }

    public IReadOnlyList<string> HtmlSequence { get; }

    public IReadOnlyList<TableFormerNormalizedBoundingBox> RawBoundingBoxes { get; }

    public IReadOnlyList<TableFormerNormalizedBoundingBox> FinalBoundingBoxes { get; }

    public bool BoundingBoxesSynced { get; }
}
