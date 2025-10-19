using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace TableFormerTorchSharpSdk.Matching;

public sealed class TableFormerMatchingPostProcessor
{
    public TableFormerMatchingDetails Process(
        MutableTableFormerMatchingDetails mutableDetails,
        bool correctOverlappingCells)
    {
        ArgumentNullException.ThrowIfNull(mutableDetails);

        var tableCells = mutableDetails.TableCells.Select(cell => cell.Clone()).ToList();
        var pdfCells = ClearPdfCells(mutableDetails.PdfCells);
        var matches = CloneMatches(mutableDetails.Matches);

        if (matches.Count == 0 && pdfCells.Count > 0)
        {
            matches = ComputeIntersectionMatches(tableCells, pdfCells);
        }

        var (columns, _, maxCellId) = GetTableDimension(tableCells);
        var adjustedCells = new List<TableFormerMutableTableCell>();

        for (var column = 0; column < columns; column++)
        {
            var (goodCells, badCells) = GetGoodBadCellsInColumn(tableCells, column, matches);
            var alignment = FindAlignmentInColumn(goodCells);
            var (position, _, width, height) = GetMedianPositionAndSize(goodCells, alignment);
            var adjustedBadCells = MoveCellsToAlignment(badCells, position, alignment, width, height);

            adjustedCells.AddRange(goodCells.Select(cell => cell.Clone()));
            adjustedCells.AddRange(adjustedBadCells);
        }

        adjustedCells.Sort((left, right) => left.CellId.CompareTo(right.CellId));
        matches = ComputeIntersectionMatches(adjustedCells, pdfCells);

        var (deduplicatedCells, deduplicatedMatches, _) = DeduplicateColumns(
            columns,
            adjustedCells,
            matches,
            matches);

        var finalMatches = DoFinalAssignment(deduplicatedMatches);
        var alignedCells = AlignTableCellsToPdf(deduplicatedCells, pdfCells, finalMatches);
        var (postMatches, postCells, _) = AssignOrphanPdfCells(alignedCells, pdfCells, finalMatches, maxCellId);

        if (correctOverlappingCells)
        {
            postCells = EnsureNonOverlapping(postCells);
        }

        var updatedDetails = new MutableTableFormerMatchingDetails(
            mutableDetails.IouThreshold,
            mutableDetails.TableBoundingBox,
            mutableDetails.PredictionBoundingBoxesPage,
            postCells,
            pdfCells,
            ToInterfaceDictionary(postMatches),
            mutableDetails.PageWidth,
            mutableDetails.PageHeight);

        return TableFormerMatchingDetails.FromMutable(updatedDetails);
    }

    private static (int Columns, int Rows, int MaxCellId) GetTableDimension(IEnumerable<TableFormerMutableTableCell> tableCells)
    {
        var columns = 1;
        var rows = 1;
        var maxCellId = 0;

        foreach (var cell in tableCells)
        {
            columns = Math.Max(columns, cell.ColumnId);
            rows = Math.Max(rows, cell.RowId);
            maxCellId = Math.Max(maxCellId, cell.CellId);
        }

        return (columns + 1, rows + 1, maxCellId);
    }

    private static (List<TableFormerMutableTableCell> GoodCells, List<TableFormerMutableTableCell> BadCells) GetGoodBadCellsInColumn(
        IEnumerable<TableFormerMutableTableCell> tableCells,
        int column,
        Dictionary<string, List<TableFormerMutableMatch>> matches)
    {
        var goodCells = new List<TableFormerMutableTableCell>();
        var badCells = new List<TableFormerMutableTableCell>();

        foreach (var cell in tableCells)
        {
            if (cell.ColumnId != column)
            {
                continue;
            }

            if (cell.CellClass <= 1)
            {
                badCells.Add(cell.Clone());
                continue;
            }

            var matched = matches.Values.Any(list => list.Any(match => match.TableCellId == cell.CellId));
            if (matched)
            {
                goodCells.Add(cell.Clone());
            }
            else
            {
                badCells.Add(cell.Clone());
            }
        }

        return (goodCells, badCells);
    }

    private static string FindAlignmentInColumn(IEnumerable<TableFormerMutableTableCell> cells)
    {
        var list = cells.ToList();
        if (list.Count == 0)
        {
            return "left";
        }

        var leftRange = AlignmentRange(list, bbox => bbox[0]);
        var centerRange = AlignmentRange(list, bbox => (bbox[0] + bbox[2]) / 2.0);
        var rightRange = AlignmentRange(list, bbox => bbox[2]);

        var alignment = "left";
        var minRange = leftRange;
        if (centerRange < minRange)
        {
            alignment = "center";
            minRange = centerRange;
        }

        if (rightRange < minRange)
        {
            alignment = "right";
        }

        return alignment;
    }

    private static double AlignmentRange(IEnumerable<TableFormerMutableTableCell> cells, Func<double[], double> selector)
    {
        var values = cells.Select(cell => selector(cell.BoundingBox)).ToArray();
        if (values.Length == 0)
        {
            return double.MaxValue;
        }

        return values.Max() - values.Min();
    }

    private static (double Position, double Vertical, double Width, double Height) GetMedianPositionAndSize(
        IEnumerable<TableFormerMutableTableCell> cells,
        string alignment)
    {
        var list = cells.ToList();
        if (list.Count == 0)
        {
            return (0.0, 0.0, 0.0, 0.0);
        }

        var positions = new List<double>();
        var verticals = new List<double>();
        var widths = new List<double>();
        var heights = new List<double>();

        foreach (var cell in list)
        {
            var bbox = cell.BoundingBox;
            positions.Add(alignment switch
            {
                "center" => (bbox[0] + bbox[2]) / 2.0,
                "right" => bbox[2],
                _ => bbox[0],
            });
            verticals.Add((bbox[1] + bbox[3]) / 2.0);
            widths.Add(Math.Abs(bbox[2] - bbox[0]));
            heights.Add(Math.Abs(bbox[3] - bbox[1]));
        }

        return (
            Median(positions),
            Median(verticals),
            Median(widths),
            Median(heights));
    }

    private static double Median(IReadOnlyList<double> values)
    {
        if (values.Count == 0)
        {
            return 0.0;
        }

        var sorted = values.OrderBy(v => v).ToArray();
        var mid = sorted.Length / 2;
        return sorted.Length % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) / 2.0
            : sorted[mid];
    }

    private static List<TableFormerMutableTableCell> MoveCellsToAlignment(
        IEnumerable<TableFormerMutableTableCell> cells,
        double position,
        string alignment,
        double width,
        double height)
    {
        var adjusted = new List<TableFormerMutableTableCell>();
        foreach (var cell in cells)
        {
            var clone = cell.Clone();
            var bbox = clone.BoundingBox;
            var centerY = (bbox[1] + bbox[3]) / 2.0;

            double left;
            double right;
            switch (alignment)
            {
                case "center":
                    left = position - width / 2.0;
                    right = position + width / 2.0;
                    break;
                case "right":
                    right = position;
                    left = right - width;
                    break;
                default:
                    left = position;
                    right = left + width;
                    break;
            }

            var top = centerY - height / 2.0;
            var bottom = centerY + height / 2.0;
            clone.BoundingBox = new[] { left, top, right, bottom };
            adjusted.Add(clone);
        }

        return adjusted;
    }

    private static Dictionary<string, List<TableFormerMutableMatch>> ComputeIntersectionMatches(
        IEnumerable<TableFormerMutableTableCell> tableCells,
        IEnumerable<TableFormerMutablePdfCell> pdfCells)
    {
        var matches = new Dictionary<string, List<TableFormerMutableMatch>>(StringComparer.Ordinal);
        var tableList = tableCells.ToList();
        var pdfList = pdfCells.ToList();

        if (tableList.Count == 0 || pdfList.Count == 0)
        {
            return matches;
        }

        var pdfAreas = pdfList.ToDictionary(
            pdf => pdf.Id,
            pdf => ComputeArea(pdf.BoundingBox.ToArray()),
            StringComparer.Ordinal);

        foreach (var cell in tableList)
        {
            var cellBbox = cell.BoundingBox;
            foreach (var pdfCell in pdfList)
            {
                var pdfBbox = pdfCell.BoundingBox.ToArray();
                var intersection = FindIntersection(cellBbox, pdfBbox);
                if (intersection is null)
                {
                    continue;
                }

                if (!pdfAreas.TryGetValue(pdfCell.Id, out var pdfArea) || pdfArea <= 0)
                {
                    continue;
                }

                var intersectionArea = ComputeArea(intersection);
                var score = intersectionArea / pdfArea;
                if (score <= 0)
                {
                    continue;
                }

                if (!matches.TryGetValue(pdfCell.Id, out var list))
                {
                    list = new List<TableFormerMutableMatch>();
                    matches[pdfCell.Id] = list;
                }

                list.Add(new TableFormerMutableMatch(cell.CellId, score, null, null));
            }
        }

        foreach (var key in matches.Keys.ToList())
        {
            matches[key] = matches[key]
                .OrderByDescending(match => match.IntersectionOverPdf ?? 0.0)
                .ToList();
        }

        return matches;
    }

    private static double ComputeArea(IReadOnlyList<double> bbox)
    {
        var width = bbox[2] - bbox[0];
        var height = bbox[3] - bbox[1];
        if (width <= 0 || height <= 0)
        {
            return 0.0;
        }

        return width * height;
    }

    private static double[]? FindIntersection(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        var left = Math.Max(a[0], b[0]);
        var top = Math.Max(a[1], b[1]);
        var right = Math.Min(a[2], b[2]);
        var bottom = Math.Min(a[3], b[3]);

        if (right <= left || bottom <= top)
        {
            return null;
        }

        return new[] { left, top, right, bottom };
    }

    private static (List<TableFormerMutableTableCell> Cells, Dictionary<string, List<TableFormerMutableMatch>> Matches, int Columns)
        DeduplicateColumns(
            int tabColumns,
            IEnumerable<TableFormerMutableTableCell> tableCells,
            Dictionary<string, List<TableFormerMutableMatch>> iouMatches,
            Dictionary<string, List<TableFormerMutableMatch>> iocMatches)
    {
        var tableList = tableCells.ToList();
        var pdfCellsInColumns = new List<List<int>>(tabColumns);
        var totalScores = new List<double>(tabColumns);

        for (var column = 0; column < tabColumns; column++)
        {
            var columnCellIds = tableList
                .Where(cell => cell.ColumnId == column)
                .Select(cell => cell.CellId)
                .ToHashSet();

            var pdfIds = new HashSet<int>();
            double score = 0.0;

            foreach (var pair in iouMatches)
            {
                foreach (var match in pair.Value)
                {
                    if (!columnCellIds.Contains(match.TableCellId))
                    {
                        continue;
                    }

                    if (match.IntersectionOverUnion.HasValue)
                    {
                        score += match.IntersectionOverUnion.Value;
                    }
                    else if (match.IntersectionOverPdf.HasValue)
                    {
                        score += match.IntersectionOverPdf.Value;
                    }

                    if (int.TryParse(pair.Key, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                    {
                        pdfIds.Add(parsed);
                    }
                }
            }

            foreach (var pair in iocMatches)
            {
                foreach (var match in pair.Value)
                {
                    if (!columnCellIds.Contains(match.TableCellId))
                    {
                        continue;
                    }

                    if (match.IntersectionOverPdf.HasValue)
                    {
                        score += match.IntersectionOverPdf.Value;
                    }

                    if (int.TryParse(pair.Key, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                    {
                        pdfIds.Add(parsed);
                    }
                }
            }

            pdfCellsInColumns.Add(pdfIds.ToList());
            totalScores.Add(score);
        }

        var columnsToRemove = new HashSet<int>();
        for (var column = 0; column < tabColumns - 1; column++)
        {
            var current = pdfCellsInColumns[column];
            if (current.Count == 0)
            {
                continue;
            }

            var next = pdfCellsInColumns[column + 1];
            var intersection = current.Intersect(next).Count();
            var overlap = intersection / (double)current.Count;
            if (overlap <= 0.6)
            {
                continue;
            }

            if (totalScores[column] >= totalScores[column + 1])
            {
                columnsToRemove.Add(column + 1);
            }
            else
            {
                columnsToRemove.Add(column);
            }
        }

        var filteredCells = new List<TableFormerMutableTableCell>();
        var removedIds = new HashSet<int>();
        foreach (var cell in tableList)
        {
            if (columnsToRemove.Contains(cell.ColumnId))
            {
                removedIds.Add(cell.CellId);
                continue;
            }

            filteredCells.Add(cell.Clone());
        }

        var filteredMatches = new Dictionary<string, List<TableFormerMutableMatch>>(StringComparer.Ordinal);
        foreach (var pair in iocMatches)
        {
            var filtered = pair.Value
                .Where(match => !removedIds.Contains(match.TableCellId))
                .Select(match => match.Clone())
                .ToList();

            if (filtered.Count > 0)
            {
                filteredMatches[pair.Key] = filtered;
            }
        }

        return (filteredCells, filteredMatches, tabColumns - columnsToRemove.Count);
    }

    private static Dictionary<string, List<TableFormerMutableMatch>> DoFinalAssignment(
        Dictionary<string, List<TableFormerMutableMatch>> matches)
    {
        var result = new Dictionary<string, List<TableFormerMutableMatch>>(StringComparer.Ordinal);
        foreach (var pair in matches)
        {
            if (pair.Value.Count == 0)
            {
                continue;
            }

            var best = pair.Value.OrderByDescending(match => match.IntersectionOverPdf ?? 0.0).First();
            result[pair.Key] = new List<TableFormerMutableMatch> { best.Clone() };
        }

        return result;
    }

    private static List<TableFormerMutableTableCell> AlignTableCellsToPdf(
        IEnumerable<TableFormerMutableTableCell> tableCells,
        IEnumerable<TableFormerMutablePdfCell> pdfCells,
        Dictionary<string, List<TableFormerMutableMatch>> matches)
    {
        var tableList = tableCells.ToList();
        var pdfLookup = pdfCells.ToDictionary(cell => cell.Id, cell => cell.BoundingBox.ToArray(), StringComparer.Ordinal);
        var aligned = new List<TableFormerMutableTableCell>(tableList.Count);

        foreach (var cell in tableList)
        {
            var clone = cell.Clone();
            foreach (var pair in matches)
            {
                if (!pair.Value.Any(match => match.TableCellId == cell.CellId))
                {
                    continue;
                }

                if (!pdfLookup.TryGetValue(pair.Key, out var bbox))
                {
                    continue;
                }

                clone.BoundingBox = (double[])bbox.Clone();
                break;
            }

            aligned.Add(clone);
        }

        return aligned;
    }

    private static (Dictionary<string, List<TableFormerMutableMatch>> Matches, List<TableFormerMutableTableCell> Cells, int MaxCellId)
        AssignOrphanPdfCells(
            IEnumerable<TableFormerMutableTableCell> tableCells,
            IEnumerable<TableFormerMutablePdfCell> pdfCells,
            Dictionary<string, List<TableFormerMutableMatch>> matches,
            int maxCellId)
    {
        var resultMatches = CloneMatches(ToInterfaceDictionary(matches));
        var resultCells = tableCells.Select(cell => cell.Clone()).ToList();
        var pdfList = pdfCells.ToList();
        var matchedPdfIds = new HashSet<string>(matches.Keys, StringComparer.Ordinal);

        foreach (var pdfCell in pdfList)
        {
            if (matchedPdfIds.Contains(pdfCell.Id))
            {
                continue;
            }

            maxCellId += 1;
            var newCell = new TableFormerMutableTableCell(
                maxCellId,
                0,
                0,
                pdfCell.BoundingBox.ToArray(),
                2,
                "body",
                string.Empty,
                null,
                null);
            resultCells.Add(newCell);
            resultMatches[pdfCell.Id] = new List<TableFormerMutableMatch>
            {
                new TableFormerMutableMatch(newCell.CellId, 1.0, null, null),
            };
        }

        return (resultMatches, resultCells, maxCellId);
    }

    private static List<TableFormerMutableTableCell> EnsureNonOverlapping(IEnumerable<TableFormerMutableTableCell> tableCells)
    {
        return tableCells.Select(cell => cell.Clone()).ToList();
    }

    private static List<TableFormerMutablePdfCell> ClearPdfCells(IEnumerable<TableFormerMutablePdfCell> pdfCells)
    {
        return pdfCells
            .Where(cell => !string.IsNullOrWhiteSpace(cell.Text))
            .Select(cell => cell.Clone())
            .ToList();
    }

    private static Dictionary<string, List<TableFormerMutableMatch>> CloneMatches(
        IEnumerable<KeyValuePair<string, IList<TableFormerMutableMatch>>> matches)
    {
        return matches.ToDictionary(
            pair => pair.Key,
            pair => pair.Value.Select(match => match.Clone()).ToList(),
            StringComparer.Ordinal);
    }

    private static Dictionary<string, IList<TableFormerMutableMatch>> ToInterfaceDictionary(
        Dictionary<string, List<TableFormerMutableMatch>> matches)
    {
        return matches.ToDictionary(
            pair => pair.Key,
            pair => (IList<TableFormerMutableMatch>)pair.Value,
            StringComparer.Ordinal);
    }
}
