using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;

namespace TableFormerSdk.Tests;

internal sealed class TableFormerCellMatchingReference
{
    private TableFormerCellMatchingReference(IReadOnlyList<TableFormerCellMatchingSample> samples)
    {
        Samples = samples;
        SamplesByKey = samples.ToDictionary(
            sample => (sample.ImageName, sample.TableIndex),
            sample => sample);
    }

    public IReadOnlyList<TableFormerCellMatchingSample> Samples { get; }

    public IReadOnlyDictionary<(string ImageName, int TableIndex), TableFormerCellMatchingSample> SamplesByKey { get; }

    public static TableFormerCellMatchingReference Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        if (!root.TryGetProperty("samples", out var samplesElement))
        {
            throw new InvalidDataException("Cell matching reference JSON is missing the 'samples' array.");
        }

        var samples = new List<TableFormerCellMatchingSample>();
        foreach (var element in samplesElement.EnumerateArray())
        {
            samples.Add(TableFormerCellMatchingSample.FromJson(element));
        }

        return new TableFormerCellMatchingReference(samples.AsReadOnly());
    }
}

internal sealed class TableFormerCellMatchingSample
{
    private TableFormerCellMatchingSample(
        string imageName,
        int tableIndex,
        string tensorSha256,
        string tagSequenceSha256,
        IReadOnlyList<double> tableBoundingBox,
        double? pageWidth,
        double? pageHeight,
        double? iouThreshold,
        IReadOnlyList<int> predictionShape,
        float[] predictionBoundingBoxes,
        string predictionSha256,
        IReadOnlyList<TableFormerCellReference> tableCells,
        string tableCellsSha256,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> matches,
        string matchesSha256,
        IReadOnlyList<TableFormerPdfCellReference> pdfCells,
        string pdfCellsSha256)
    {
        ImageName = imageName;
        TableIndex = tableIndex;
        TensorSha256 = tensorSha256;
        TagSequenceSha256 = tagSequenceSha256;
        TableBoundingBox = tableBoundingBox;
        PageWidth = pageWidth;
        PageHeight = pageHeight;
        IouThreshold = iouThreshold;
        PredictionShape = predictionShape;
        PredictionBoundingBoxes = predictionBoundingBoxes;
        PredictionSha256 = predictionSha256;
        TableCells = tableCells;
        TableCellsSha256 = tableCellsSha256;
        Matches = matches;
        MatchesSha256 = matchesSha256;
        PdfCells = pdfCells;
        PdfCellsSha256 = pdfCellsSha256;
    }

    public string ImageName { get; }

    public int TableIndex { get; }

    public string TensorSha256 { get; }

    public string TagSequenceSha256 { get; }

    public IReadOnlyList<double> TableBoundingBox { get; }

    public double? PageWidth { get; }

    public double? PageHeight { get; }

    public double? IouThreshold { get; }

    public IReadOnlyList<int> PredictionShape { get; }

    public float[] PredictionBoundingBoxes { get; }

    public string PredictionSha256 { get; }

    public IReadOnlyList<TableFormerCellReference> TableCells { get; }

    public string TableCellsSha256 { get; }

    public IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> Matches { get; }

    public string MatchesSha256 { get; }

    public IReadOnlyList<TableFormerPdfCellReference> PdfCells { get; }

    public string PdfCellsSha256 { get; }

    public static TableFormerCellMatchingSample FromJson(JsonElement element)
    {
        var imageName = element.GetProperty("image_name").GetString()
            ?? throw new InvalidDataException("Cell matching sample missing 'image_name'.");
        var tableIndex = element.GetProperty("table_index").GetInt32();
        var tensorSha256 = element.GetProperty("tensor_sha256").GetString()
            ?? throw new InvalidDataException($"Sample '{imageName}' missing 'tensor_sha256'.");
        var tagSha256 = element.GetProperty("tag_sequence_sha256").GetString()
            ?? throw new InvalidDataException($"Sample '{imageName}' missing 'tag_sequence_sha256'.");

        var tableBoundingBox = element.GetProperty("table_bbox")
            .EnumerateArray()
            .Select(v => v.GetDouble())
            .ToArray();

        var predictionShape = element.GetProperty("prediction_bbox_shape")
            .EnumerateArray()
            .Select(v => v.GetInt32())
            .ToArray();

        var predictionBase64 = element.GetProperty("prediction_bbox_zlib_base64").GetString()
            ?? throw new InvalidDataException($"Sample '{imageName}' missing prediction bbox payload.");
        var predictionBoundingBoxes = DecodeFloatArray(predictionBase64, predictionShape);

        var tableCells = ParseTableCells(element.GetProperty("table_cells"));
        var matches = ParseMatches(element.GetProperty("matches"));
        var pdfCells = ParsePdfCells(element.GetProperty("pdf_cells"));

        return new TableFormerCellMatchingSample(
            imageName,
            tableIndex,
            tensorSha256,
            tagSha256,
            new ReadOnlyCollection<double>(tableBoundingBox),
            element.TryGetProperty("page_width", out var pageWidthElement) ? pageWidthElement.GetDouble() : null,
            element.TryGetProperty("page_height", out var pageHeightElement) ? pageHeightElement.GetDouble() : null,
            element.TryGetProperty("iou_threshold", out var iouElement) ? iouElement.GetDouble() : null,
            new ReadOnlyCollection<int>(predictionShape),
            predictionBoundingBoxes,
            element.GetProperty("prediction_bbox_sha256").GetString()
                ?? throw new InvalidDataException($"Sample '{imageName}' missing 'prediction_bbox_sha256'."),
            tableCells,
            element.GetProperty("table_cells_sha256").GetString()
                ?? throw new InvalidDataException($"Sample '{imageName}' missing 'table_cells_sha256'."),
            matches,
            element.GetProperty("matches_sha256").GetString()
                ?? throw new InvalidDataException($"Sample '{imageName}' missing 'matches_sha256'."),
            pdfCells,
            element.GetProperty("pdf_cells_sha256").GetString()
                ?? throw new InvalidDataException($"Sample '{imageName}' missing 'pdf_cells_sha256'."));
    }

    private static IReadOnlyList<TableFormerCellReference> ParseTableCells(JsonElement element)
    {
        var cells = new List<TableFormerCellReference>();
        foreach (var cellElement in element.EnumerateArray())
        {
            var cellId = cellElement.GetProperty("cell_id").GetInt32();
            var rowId = cellElement.GetProperty("row_id").GetInt32();
            var columnId = cellElement.GetProperty("column_id").GetInt32();
            var bbox = cellElement.GetProperty("bbox").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            var cellClass = cellElement.GetProperty("cell_class").GetInt32();
            var label = cellElement.GetProperty("label").GetString() ?? string.Empty;
            var multicolTag = cellElement.TryGetProperty("multicol_tag", out var multicolElement)
                ? multicolElement.GetString() ?? string.Empty
                : string.Empty;
            int? colspan = null;
            if (cellElement.TryGetProperty("colspan", out var colspanElement))
            {
                colspan = colspanElement.GetInt32();
            }

            int? rowspan = null;
            if (cellElement.TryGetProperty("rowspan", out var rowspanElement))
            {
                rowspan = rowspanElement.GetInt32();
            }

            int? colspanValue = null;
            if (cellElement.TryGetProperty("colspan_val", out var colspanValElement))
            {
                colspanValue = colspanValElement.GetInt32();
            }

            int? rowspanValue = null;
            if (cellElement.TryGetProperty("rowspan_val", out var rowspanValElement))
            {
                rowspanValue = rowspanValElement.GetInt32();
            }

            cells.Add(new TableFormerCellReference(
                cellId,
                rowId,
                columnId,
                new ReadOnlyCollection<double>(bbox),
                cellClass,
                label,
                multicolTag,
                colspan,
                rowspan,
                colspanValue,
                rowspanValue));
        }

        return new ReadOnlyCollection<TableFormerCellReference>(cells);
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> ParseMatches(JsonElement element)
    {
        var matches = new Dictionary<string, IReadOnlyList<TableFormerMatchReference>>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            var entries = new List<TableFormerMatchReference>();
            foreach (var matchElement in property.Value.EnumerateArray())
            {
                var tableCellId = matchElement.GetProperty("table_cell_id").GetInt32();
                double? intersectionOverPdf = null;
                if (matchElement.TryGetProperty("iopdf", out var iopdfElement))
                {
                    intersectionOverPdf = iopdfElement.GetDouble();
                }

                double? intersectionOverUnion = null;
                if (matchElement.TryGetProperty("iou", out var iouElement))
                {
                    intersectionOverUnion = iouElement.GetDouble();
                }

                double? postScore = null;
                if (matchElement.TryGetProperty("post", out var postElement))
                {
                    postScore = postElement.GetDouble();
                }

                entries.Add(new TableFormerMatchReference(
                    tableCellId,
                    intersectionOverPdf,
                    intersectionOverUnion,
                    postScore));
            }

            matches[property.Name] = new ReadOnlyCollection<TableFormerMatchReference>(entries);
        }

        return new ReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>>(matches);
    }

    private static IReadOnlyList<TableFormerPdfCellReference> ParsePdfCells(JsonElement element)
    {
        var pdfCells = new List<TableFormerPdfCellReference>();
        foreach (var cellElement in element.EnumerateArray())
        {
            var id = cellElement.TryGetProperty("id", out var idElement) ? idElement.GetString() ?? string.Empty : string.Empty;
            var text = cellElement.TryGetProperty("text", out var textElement) ? textElement.GetString() ?? string.Empty : string.Empty;
            var bbox = cellElement.GetProperty("bbox").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            pdfCells.Add(new TableFormerPdfCellReference(id, text, new ReadOnlyCollection<double>(bbox)));
        }

        return new ReadOnlyCollection<TableFormerPdfCellReference>(pdfCells);
    }

    private static float[] DecodeFloatArray(string base64, IReadOnlyList<int> shape)
    {
        var expectedCount = shape.Aggregate(1, (product, dimension) => product * dimension);
        var compressed = Convert.FromBase64String(base64);
        using var input = new MemoryStream(compressed);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);

        var bytes = output.ToArray();
        if (bytes.Length != expectedCount * sizeof(float))
        {
            throw new InvalidDataException(
                $"Decoded float array length {bytes.Length} does not match expected {expectedCount * sizeof(float)} bytes.");
        }

        var result = new float[expectedCount];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }
}

internal sealed record TableFormerCellReference(
    int CellId,
    int RowId,
    int ColumnId,
    IReadOnlyList<double> BoundingBox,
    int CellClass,
    string Label,
    string? MulticolTag,
    int? Colspan,
    int? Rowspan,
    int? ColspanValue,
    int? RowspanValue);

internal sealed record TableFormerMatchReference(
    int TableCellId,
    double? IntersectionOverPdf,
    double? IntersectionOverUnion,
    double? PostScore);

internal sealed record TableFormerPdfCellReference(string Id, string Text, IReadOnlyList<double> BoundingBox);
