using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.PagePreparation;

namespace TableFormerTorchSharpSdk.Results;

public sealed class TableFormerDoclingResponseAssembler
{
    public TableFormerDoclingTablePrediction Assemble(
        TableFormerMatchingDetails matchingDetails,
        TableFormerSequencePrediction sequencePrediction,
        bool sortRowColIndexes = true)
    {
        ArgumentNullException.ThrowIfNull(matchingDetails);
        ArgumentNullException.ThrowIfNull(sequencePrediction);

        var tableCells = matchingDetails.TableCells;
        var matches = matchingDetails.Matches;
        var pdfCells = matchingDetails.PdfCells;
        var boundingBoxes = BuildBoundingBoxMap(matchingDetails.PredictionBoundingBoxesPage);

        var doclingOutput = GenerateDoclingResponses(tableCells, matches, boundingBoxes);
        List<TableFormerDoclingCellResponse> doclingResponses;
        List<TableFormerDoclingCellResponse> tfResponses;

        if (doclingOutput.Count > 0)
        {
            doclingOutput.Sort((left, right) => left.CellId.CompareTo(right.CellId));
            doclingResponses = doclingOutput;
            tfResponses = MergeTfOutput(doclingOutput, pdfCells);
        }
        else
        {
            tfResponses = GenerateDummyResponses(tableCells, boundingBoxes);
            tfResponses.Sort((left, right) => left.CellId.CompareTo(right.CellId));
            doclingResponses = tfResponses;
        }

        int numCols;
        int numRows;
        if (sortRowColIndexes)
        {
            (numCols, numRows) = AdjustOffsetsAndComputeDimensions(tfResponses);
        }
        else
        {
            (numCols, numRows) = ComputeDimensionsFromSequence(sequencePrediction.RsSequence);
        }

        var prediction = new TableFormerDoclingPrediction(
            sequencePrediction.FinalBoundingBoxes,
            sequencePrediction.ClassPredictions,
            sequencePrediction.TagSequence,
            sequencePrediction.RsSequence,
            sequencePrediction.HtmlSequence);

        var predictDetails = new TableFormerDoclingPredictDetails(
            matchingDetails.IouThreshold,
            matchingDetails.TableBoundingBox,
            matchingDetails.PredictionBoundingBoxesPage,
            prediction,
            matchingDetails.PdfCells,
            matchingDetails.PageWidth,
            matchingDetails.PageHeight,
            matchingDetails.TableCells,
            matchingDetails.Matches,
            doclingResponses,
            numCols,
            numRows);

        return new TableFormerDoclingTablePrediction(tfResponses, predictDetails);
    }

    private static Dictionary<int, TableFormerBoundingBox> BuildBoundingBoxMap(
        IReadOnlyList<TableFormerBoundingBox> boundingBoxes)
    {
        var map = new Dictionary<int, TableFormerBoundingBox>(boundingBoxes.Count);
        for (var i = 0; i < boundingBoxes.Count; i++)
        {
            map[i] = boundingBoxes[i];
        }

        return map;
    }

    private static List<TableFormerDoclingCellResponse> GenerateDoclingResponses(
        IReadOnlyList<TableFormerMutableTableCell> tableCells,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>> matches,
        IReadOnlyDictionary<int, TableFormerBoundingBox> boundingBoxes)
    {
        var responses = new List<TableFormerDoclingCellResponse>();
        if (matches.Count == 0)
        {
            return responses;
        }

        var lookup = tableCells.ToDictionary(cell => cell.CellId);
        foreach (var pair in matches)
        {
            var cellId = ParsePdfCellId(pair.Key);
            var response = new TableFormerDoclingCellResponse(cellId);
            foreach (var match in pair.Value)
            {
                if (!lookup.TryGetValue(match.TableCellId, out var tableCell))
                {
                    continue;
                }

                response.BoundingBox = TryResolveBoundingBox(
                    match.TableCellId,
                    tableCell,
                    boundingBoxes);

                var colSpan = tableCell.ColspanValue ?? tableCell.Colspan ?? 1;
                var rowSpan = tableCell.RowspanValue ?? tableCell.Rowspan ?? 1;

                response.ColSpan = colSpan;
                response.RowSpan = rowSpan;
                response.StartColOffsetIndex = tableCell.ColumnId;
                response.EndColOffsetIndex = tableCell.ColumnId + colSpan;
                response.StartRowOffsetIndex = tableCell.RowId;
                response.EndRowOffsetIndex = tableCell.RowId + rowSpan;

                var label = tableCell.Label ?? string.Empty;
                if (string.Equals(label, "ched", StringComparison.Ordinal))
                {
                    response.ColumnHeader = true;
                }

                if (string.Equals(label, "rhed", StringComparison.Ordinal))
                {
                    response.RowHeader = true;
                }

                if (string.Equals(label, "srow", StringComparison.Ordinal))
                {
                    response.RowSection = true;
                }
            }

            responses.Add(response);
        }

        return responses;
    }

    private static List<TableFormerDoclingCellResponse> GenerateDummyResponses(
        IReadOnlyList<TableFormerMutableTableCell> tableCells,
        IReadOnlyDictionary<int, TableFormerBoundingBox> boundingBoxes)
    {
        var responses = new List<TableFormerDoclingCellResponse>();
        if (tableCells.Count == 0)
        {
            return responses;
        }

        foreach (var tableCell in tableCells)
        {
            var response = new TableFormerDoclingCellResponse(tableCell.CellId)
            {
                BoundingBox = TryResolveBoundingBox(tableCell.CellId, tableCell, boundingBoxes),
            };

            var colSpan = tableCell.ColspanValue ?? tableCell.Colspan ?? 1;
            var rowSpan = tableCell.RowspanValue ?? tableCell.Rowspan ?? 1;

            response.ColSpan = colSpan;
            response.RowSpan = rowSpan;
            response.StartColOffsetIndex = tableCell.ColumnId;
            response.EndColOffsetIndex = tableCell.ColumnId + colSpan;
            response.StartRowOffsetIndex = tableCell.RowId;
            response.EndRowOffsetIndex = tableCell.RowId + rowSpan;

            var label = tableCell.Label ?? string.Empty;
            response.ColumnHeader = string.Equals(label, "ched", StringComparison.Ordinal);
            response.RowHeader = string.Equals(label, "rhed", StringComparison.Ordinal);
            response.RowSection = string.Equals(label, "srow", StringComparison.Ordinal);

            responses.Add(response);
        }

        return responses;
    }

    private static TableFormerDoclingBoundingBox TryResolveBoundingBox(
        int cellId,
        TableFormerMutableTableCell tableCell,
        IReadOnlyDictionary<int, TableFormerBoundingBox> boundingBoxes)
    {
        var bbox = tableCell.BoundingBox;
        if (bbox is { Length: 4 })
        {
            return TableFormerDoclingBoundingBox.FromArray(bbox, string.Empty);
        }

        if (boundingBoxes.TryGetValue(cellId, out var mapped))
        {
            return TableFormerDoclingBoundingBox.FromBoundingBox(mapped, string.Empty);
        }

        return TableFormerDoclingBoundingBox.FromArray(new[] { 0.0, 0.0, 0.0, 0.0 }, string.Empty);
    }

    private static List<TableFormerDoclingCellResponse> MergeTfOutput(
        IReadOnlyList<TableFormerDoclingCellResponse> doclingOutput,
        IReadOnlyList<TableFormerMutablePdfCell> pdfCells)
    {
        var tfOutput = new List<TableFormerDoclingCellResponse>();
        var map = new Dictionary<string, TableFormerDoclingCellResponse>(StringComparer.Ordinal);

        foreach (var doclingCell in doclingOutput)
        {
            var key = CreateCellKey(doclingCell.StartColOffsetIndex, doclingCell.StartRowOffsetIndex);
            if (map.TryGetValue(key, out var existing))
            {
                AppendPdfBoundingBoxes(existing, doclingCell.CellId, pdfCells);
                continue;
            }

            var clone = doclingCell.CloneWithoutTextCells();
            AppendPdfBoundingBoxes(clone, doclingCell.CellId, pdfCells);
            map[key] = clone;
            tfOutput.Add(clone);
        }

        return tfOutput;
    }

    private static void AppendPdfBoundingBoxes(
        TableFormerDoclingCellResponse response,
        int doclingCellId,
        IReadOnlyList<TableFormerMutablePdfCell> pdfCells)
    {
        if (pdfCells.Count == 0)
        {
            return;
        }

        var target = doclingCellId.ToString(CultureInfo.InvariantCulture);
        foreach (var pdfCell in pdfCells)
        {
            if (!IsMatchingPdfCell(pdfCell.Id, doclingCellId, target))
            {
                continue;
            }

            var bbox = TableFormerDoclingBoundingBox.FromBoundingBox(pdfCell.BoundingBox, pdfCell.Text);
            response.AddTextCellBoundingBox(bbox);
        }
    }

    private static bool IsMatchingPdfCell(string pdfCellId, int doclingCellId, string doclingCellIdString)
    {
        if (string.IsNullOrEmpty(pdfCellId))
        {
            return false;
        }

        if (int.TryParse(pdfCellId, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
        {
            return parsed == doclingCellId;
        }

        return string.Equals(pdfCellId, doclingCellIdString, StringComparison.Ordinal);
    }

    private static string CreateCellKey(int startCol, int startRow)
    {
        return string.Format(CultureInfo.InvariantCulture, "{0}_{1}", startCol, startRow);
    }

    private static (int NumCols, int NumRows) AdjustOffsetsAndComputeDimensions(
        IList<TableFormerDoclingCellResponse> responses)
    {
        var startCols = new List<int>();
        var startRows = new List<int>();

        foreach (var cell in responses)
        {
            if (!startCols.Contains(cell.StartColOffsetIndex))
            {
                startCols.Add(cell.StartColOffsetIndex);
            }

            if (!startRows.Contains(cell.StartRowOffsetIndex))
            {
                startRows.Add(cell.StartRowOffsetIndex);
            }
        }

        startCols.Sort();
        startRows.Sort();

        var maxEndCol = 0;
        var maxEndRow = 0;

        foreach (var cell in responses)
        {
            var startColIndex = IndexOf(startCols, cell.StartColOffsetIndex);
            var startRowIndex = IndexOf(startRows, cell.StartRowOffsetIndex);

            cell.StartColOffsetIndex = startColIndex;
            cell.EndColOffsetIndex = startColIndex >= 0 ? startColIndex + cell.ColSpan : startColIndex;
            maxEndCol = Math.Max(maxEndCol, cell.EndColOffsetIndex);

            cell.StartRowOffsetIndex = startRowIndex;
            cell.EndRowOffsetIndex = startRowIndex >= 0 ? startRowIndex + cell.RowSpan : startRowIndex;
            maxEndRow = Math.Max(maxEndRow, cell.EndRowOffsetIndex);
        }

        return (maxEndCol, maxEndRow);
    }

    private static (int NumCols, int NumRows) ComputeDimensionsFromSequence(
        IReadOnlyList<string> rsSequence)
    {
        if (rsSequence.Count == 0)
        {
            return (0, 0);
        }

        var firstNl = -1;
        var rowCount = 0;
        for (var i = 0; i < rsSequence.Count; i++)
        {
            if (!string.Equals(rsSequence[i], "nl", StringComparison.Ordinal))
            {
                continue;
            }

            if (firstNl < 0)
            {
                firstNl = i;
            }

            rowCount += 1;
        }

        if (firstNl < 0)
        {
            throw new InvalidOperationException("Sequence does not contain the 'nl' token required to compute table dimensions.");
        }

        return (firstNl, rowCount);
    }

    private static int ParsePdfCellId(string pdfCellId)
    {
        try
        {
            return int.Parse(pdfCellId, NumberStyles.Integer, CultureInfo.InvariantCulture);
        }
        catch (FormatException exception)
        {
            throw new FormatException($"PDF cell identifier '{pdfCellId}' is not a valid integer.", exception);
        }
    }

    private static int IndexOf(IReadOnlyList<int> values, int value)
    {
        for (var i = 0; i < values.Count; i++)
        {
            if (values[i] == value)
            {
                return i;
            }
        }

        throw new InvalidOperationException($"Unable to locate index for value '{value}'.");
    }
}
