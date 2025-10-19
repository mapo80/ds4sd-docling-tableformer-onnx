using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;

using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.PagePreparation;

namespace TableFormerTorchSharpSdk.Matching;

public sealed class TableFormerCellMatcher
{
    private readonly double _iouThreshold;

    public TableFormerCellMatcher(TableFormerConfigSnapshot configSnapshot)
    {
        ArgumentNullException.ThrowIfNull(configSnapshot);
        _iouThreshold = ResolveIouThreshold(configSnapshot.Config);
    }

    public TableFormerCellMatchingResult MatchCells(
        TableFormerPageInputSnapshot pageSnapshot,
        TableFormerTableCropSnapshot cropSnapshot,
        TableFormerSequencePrediction sequencePrediction)
    {
        ArgumentNullException.ThrowIfNull(pageSnapshot);
        ArgumentNullException.ThrowIfNull(cropSnapshot);
        ArgumentNullException.ThrowIfNull(sequencePrediction);

        var pdfCells = ClonePdfCells(pageSnapshot.Tokens);
        var translatedBoundingBoxes = TranslateBoundingBoxes(
            cropSnapshot.OriginalBoundingBox,
            sequencePrediction.FinalBoundingBoxes);
        var tableCells = BuildTableCells(
            sequencePrediction.HtmlSequence,
            sequencePrediction.RsSequence,
            translatedBoundingBoxes,
            sequencePrediction.ClassPredictions);
        var matches = pdfCells.Count > 0
            ? ComputeMatches(tableCells, pdfCells)
            : CreateEmptyMatches();

        return new TableFormerCellMatchingResult(
            _iouThreshold,
            cropSnapshot.OriginalBoundingBox,
            translatedBoundingBoxes,
            tableCells,
            pdfCells,
            matches,
            pageSnapshot.Width,
            pageSnapshot.Height);
    }

    private static double ResolveIouThreshold(JsonObject config)
    {
        if (!config.TryGetPropertyValue("predict", out var predictNode) || predictNode is not JsonObject predictObject)
        {
            throw new InvalidDataException("Configuration missing 'predict' section required for cell matching.");
        }

        if (!predictObject.TryGetPropertyValue("pdf_cell_iou_thres", out var thresholdNode) || thresholdNode is null)
        {
            throw new InvalidDataException("Configuration missing 'predict.pdf_cell_iou_thres'.");
        }

        return thresholdNode.GetValue<double>();
    }

    private static IReadOnlyList<TableFormerPdfCell> ClonePdfCells(IReadOnlyList<TableFormerPageToken> tokens)
    {
        if (tokens.Count == 0)
        {
            return Array.Empty<TableFormerPdfCell>();
        }

        var cloned = new TableFormerPdfCell[tokens.Count];
        for (var i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            var identifier = string.IsNullOrWhiteSpace(token.Id)
                ? i.ToString(CultureInfo.InvariantCulture)
                : token.Id!;
            var text = token.Text ?? string.Empty;
            cloned[i] = new TableFormerPdfCell(identifier, token.BoundingBox, text);
        }

        return new ReadOnlyCollection<TableFormerPdfCell>(cloned);
    }

    private static IReadOnlyList<TableFormerBoundingBox> TranslateBoundingBoxes(
        TableFormerBoundingBox tableBoundingBox,
        IReadOnlyList<TableFormerNormalizedBoundingBox> normalizedBoundingBoxes)
    {
        if (normalizedBoundingBoxes.Count == 0)
        {
            return Array.Empty<TableFormerBoundingBox>();
        }

        var leftOrigin = tableBoundingBox.Left;
        var topOrigin = tableBoundingBox.Top;
        var bottomOrigin = tableBoundingBox.Bottom;
        var width = tableBoundingBox.Right - leftOrigin;
        var height = bottomOrigin - topOrigin;
        var translated = new TableFormerBoundingBox[normalizedBoundingBoxes.Count];

        for (var i = 0; i < normalizedBoundingBoxes.Count; i++)
        {
            var bbox = normalizedBoundingBoxes[i];

            var x1Prime = leftOrigin + width * bbox.Left;
            var y1Prime = bottomOrigin - height * bbox.Top;
            var x2Prime = leftOrigin + width * bbox.Right;
            var y2Prime = bottomOrigin - height * bbox.Bottom;

            var top = bottomOrigin - y1Prime + topOrigin;
            var bottom = bottomOrigin - y2Prime + topOrigin;

            translated[i] = new TableFormerBoundingBox(x1Prime, top, x2Prime, bottom);
        }

        return new ReadOnlyCollection<TableFormerBoundingBox>(translated);
    }

    private static IReadOnlyList<TableFormerTableCell> BuildTableCells(
        IReadOnlyList<string> htmlSequence,
        IReadOnlyList<string> rsSequence,
        IReadOnlyList<TableFormerBoundingBox> pageBoundingBoxes,
        IReadOnlyList<int> classPredictions)
    {
        ArgumentNullException.ThrowIfNull(htmlSequence);
        ArgumentNullException.ThrowIfNull(rsSequence);
        ArgumentNullException.ThrowIfNull(pageBoundingBoxes);
        ArgumentNullException.ThrowIfNull(classPredictions);

        _ = htmlSequence;

        var spanMap = OtslHtmlConverter.ComputeSpans(rsSequence);
        var tableCells = new List<TableFormerTableCell>();
        var cellId = 0;
        var rowId = 0;
        var columnId = 0;

        foreach (var tag in rsSequence)
        {
            if (tag == "nl")
            {
                rowId += 1;
                columnId = 0;
                continue;
            }

            if (IsCellToken(tag))
            {
                var bbox = cellId < pageBoundingBoxes.Count
                    ? pageBoundingBoxes[cellId]
                    : new TableFormerBoundingBox(0.0, 0.0, 0.0, 0.0);
                var cellClass = cellId < classPredictions.Count ? classPredictions[cellId] : 2;

                spanMap.TryGetValue(cellId, out var span);
                int? colspan = null;
                int? rowspan = null;
                if (span.Colspan > 0)
                {
                    colspan = span.Colspan;
                }

                if (span.Rowspan > 0)
                {
                    rowspan = span.Rowspan;
                }

                var tableCell = new TableFormerTableCell(
                    cellId,
                    rowId,
                    columnId,
                    bbox,
                    cellClass,
                    tag,
                    string.Empty,
                    colspan,
                    rowspan);

                tableCells.Add(tableCell);
                cellId += 1;
            }

            columnId += 1;
        }

        return new ReadOnlyCollection<TableFormerTableCell>(tableCells);
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<TableFormerCellMatch>> ComputeMatches(
        IReadOnlyList<TableFormerTableCell> tableCells,
        IReadOnlyList<TableFormerPdfCell> pdfCells)
    {
        var matches = new Dictionary<string, List<TableFormerCellMatch>>(StringComparer.Ordinal);
        var pdfAreas = pdfCells.Select(pdf => ComputeArea(pdf.BoundingBox)).ToArray();

        for (var i = 0; i < tableCells.Count; i++)
        {
            var tableCell = tableCells[i];
            for (var j = 0; j < pdfCells.Count; j++)
            {
                var pdfCell = pdfCells[j];
                var intersectionCandidate = FindIntersection(tableCell.BoundingBox, pdfCell.BoundingBox);
                if (intersectionCandidate is not TableFormerBoundingBox intersection)
                {
                    continue;
                }

                var intersectionArea = ComputeArea(intersection);
                if (intersectionArea <= 0)
                {
                    continue;
                }

                var pdfArea = pdfAreas[j];
                if (pdfArea <= 0)
                {
                    continue;
                }

                var iopdf = intersectionArea / pdfArea;
                if (iopdf <= 0)
                {
                    continue;
                }

                if (!matches.TryGetValue(pdfCell.Id, out var list))
                {
                    list = new List<TableFormerCellMatch>();
                    matches[pdfCell.Id] = list;
                }

                var match = new TableFormerCellMatch(tableCell.CellId, iopdf);
                var duplicate = list.Any(existing => existing.TableCellId == match.TableCellId
                    && Math.Abs(existing.IntersectionOverPdf - match.IntersectionOverPdf) <= 1e-9);
                if (!duplicate)
                {
                    list.Add(match);
                }
            }
        }

        return matches.Count == 0
            ? CreateEmptyMatches()
            : new ReadOnlyDictionary<string, IReadOnlyList<TableFormerCellMatch>>(
                matches.ToDictionary(
                    pair => pair.Key,
                    pair => (IReadOnlyList<TableFormerCellMatch>)new ReadOnlyCollection<TableFormerCellMatch>(pair.Value),
                    StringComparer.Ordinal));
    }

    private static double ComputeArea(TableFormerBoundingBox bbox)
    {
        var width = bbox.Right - bbox.Left;
        var height = bbox.Bottom - bbox.Top;
        if (width <= 0 || height <= 0)
        {
            return 0;
        }

        return width * height;
    }

    private static TableFormerBoundingBox? FindIntersection(
        TableFormerBoundingBox bbox1,
        TableFormerBoundingBox bbox2)
    {
        if (bbox1.Right < bbox2.Left || bbox2.Right < bbox1.Left || bbox1.Top > bbox2.Bottom || bbox2.Top > bbox2.Bottom)
        {
            return null;
        }

        var left = Math.Max(bbox1.Left, bbox2.Left);
        var top = Math.Max(bbox1.Top, bbox2.Top);
        var right = Math.Min(bbox1.Right, bbox2.Right);
        var bottom = Math.Min(bbox1.Bottom, bbox2.Bottom);

        if (right <= left || bottom <= top)
        {
            return null;
        }

        return new TableFormerBoundingBox(left, top, right, bottom);
    }

    private static bool IsCellToken(string token)
    {
        return token is "fcel" or "ecel" or "xcel" or "ched" or "rhed" or "srow";
    }

    private static ReadOnlyDictionary<string, IReadOnlyList<TableFormerCellMatch>> CreateEmptyMatches()
    {
        return new ReadOnlyDictionary<string, IReadOnlyList<TableFormerCellMatch>>(
            new Dictionary<string, IReadOnlyList<TableFormerCellMatch>>(StringComparer.Ordinal));
    }
}

public sealed class TableFormerCellMatchingResult
{
    public TableFormerCellMatchingResult(
        double iouThreshold,
        TableFormerBoundingBox tableBoundingBox,
        IReadOnlyList<TableFormerBoundingBox> predictionBoundingBoxes,
        IReadOnlyList<TableFormerTableCell> tableCells,
        IReadOnlyList<TableFormerPdfCell> pdfCells,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerCellMatch>> matches,
        int pageWidth,
        int pageHeight)
    {
        IouThreshold = iouThreshold;
        TableBoundingBox = tableBoundingBox;
        PredictionBoundingBoxesPage = predictionBoundingBoxes;
        TableCells = tableCells;
        PdfCells = pdfCells;
        Matches = matches;
        PageWidth = pageWidth;
        PageHeight = pageHeight;
    }

    public double IouThreshold { get; }

    public TableFormerBoundingBox TableBoundingBox { get; }

    public IReadOnlyList<TableFormerBoundingBox> PredictionBoundingBoxesPage { get; }

    public IReadOnlyList<TableFormerTableCell> TableCells { get; }

    public IReadOnlyList<TableFormerPdfCell> PdfCells { get; }

    public IReadOnlyDictionary<string, IReadOnlyList<TableFormerCellMatch>> Matches { get; }

    public int PageWidth { get; }

    public int PageHeight { get; }
}

public sealed record TableFormerTableCell(
    int CellId,
    int RowId,
    int ColumnId,
    TableFormerBoundingBox BoundingBox,
    int CellClass,
    string Label,
    string MulticolTag,
    int? Colspan,
    int? Rowspan);

public sealed record TableFormerPdfCell(string Id, TableFormerBoundingBox BoundingBox, string Text);

public sealed record TableFormerCellMatch(int TableCellId, double IntersectionOverPdf);
