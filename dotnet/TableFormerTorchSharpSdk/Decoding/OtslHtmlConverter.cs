using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;

namespace TableFormerTorchSharpSdk.Decoding;

internal static class OtslHtmlConverter
{
    private static readonly HashSet<string> CellTags = new(new[] { "fcel", "ecel", "ched", "rhed", "srow" });

    public static IReadOnlyList<string> ToHtml(IReadOnlyList<string> rsSequence)
    {
        ArgumentNullException.ThrowIfNull(rsSequence);

        if (rsSequence.Count == 0)
        {
            return Array.Empty<string>();
        }

        if (!CellTags.Contains(rsSequence[0]))
        {
            return new ReadOnlyCollection<string>(rsSequence.ToArray());
        }

        var workingSequence = rsSequence;
        if (!IsSquare(rsSequence))
        {
            workingSequence = PadToSquare(rsSequence, "lcel");
        }

        var rows = SplitRows(workingSequence);
        var html = new List<string>();
        var registry2dSpan = new HashSet<string>(StringComparer.Ordinal);
        var theadPresent = false;

        for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            var row = rows[rowIndex];
            var htmlList = new List<string>();

            if (!theadPresent && row.Contains("ched"))
            {
                htmlList.Add("<thead>");
                theadPresent = true;
            }

            if (theadPresent && !row.Contains("ched"))
            {
                htmlList.Add("</thead>");
                theadPresent = false;
            }

            htmlList.Add("<tr>");

            for (var cellIndex = 0; cellIndex < row.Count; cellIndex++)
            {
                var cell = row[cellIndex];
                if (!CellTags.Contains(cell))
                {
                    continue;
                }

                var rdist = 0;
                var ddist = 0;
                var xrdist = 0;
                var xddist = 0;
                var span = false;

                if (cellIndex + 1 < row.Count && row[cellIndex + 1] == "lcel")
                {
                    rdist = CheckRight(rows, cellIndex, rowIndex);
                    span = true;
                }

                if (rowIndex + 1 < rows.Count && rows[rowIndex + 1][cellIndex] == "ucel")
                {
                    ddist = CheckDown(rows, cellIndex, rowIndex);
                    span = true;
                }

                if (cellIndex + 1 < row.Count && row[cellIndex + 1] == "xcel")
                {
                    xrdist = CheckRight(rows, cellIndex, rowIndex);
                    xddist = CheckDown(rows, cellIndex, rowIndex);
                    span = true;

                    for (var x = cellIndex; x < cellIndex + xrdist; x++)
                    {
                        for (var y = rowIndex; y < rowIndex + xddist; y++)
                        {
                            var key = FormSpanKey(x, y);
                            if (registry2dSpan.Contains(key))
                            {
                                span = false;
                            }
                        }
                    }

                    if (span)
                    {
                        for (var x = cellIndex; x < cellIndex + xrdist; x++)
                        {
                            for (var y = rowIndex; y < rowIndex + xddist; y++)
                            {
                                registry2dSpan.Add(FormSpanKey(x, y));
                            }
                        }
                    }
                }

                if (span)
                {
                    htmlList.Add("<td");
                    if (rdist > 1)
                    {
                        htmlList.Add($" colspan=\"{rdist}\"");
                    }

                    if (ddist > 1)
                    {
                        htmlList.Add($" rowspan=\"{ddist}\"");
                    }

                    if (xrdist > 1)
                    {
                        htmlList.Add($" rowspan=\"{xddist}\"");
                        htmlList.Add($" colspan=\"{xrdist}\"");
                    }

                    htmlList.Add(">");
                    htmlList.Add("</td>");
                }
                else
                {
                    htmlList.Add("<td>");
                    htmlList.Add("</td>");
                }
            }

            htmlList.Add("</tr>");
            html.AddRange(htmlList);
        }

        return new ReadOnlyCollection<string>(html.ToArray());
    }

    public static IReadOnlyDictionary<int, (int Colspan, int Rowspan)> ComputeSpans(
        IReadOnlyList<string> rsSequence)
    {
        ArgumentNullException.ThrowIfNull(rsSequence);

        var spanMap = new Dictionary<int, (int Colspan, int Rowspan)>();
        if (rsSequence.Count == 0)
        {
            return spanMap;
        }

        if (!CellTags.Contains(rsSequence[0]) && rsSequence[0] != "xcel")
        {
            return spanMap;
        }

        var workingSequence = rsSequence;
        if (!IsSquare(rsSequence))
        {
            workingSequence = PadToSquare(rsSequence, "lcel");
        }

        var rows = SplitRows(workingSequence);
        if (rows.Count == 0)
        {
            return spanMap;
        }

        var registry2dSpan = new HashSet<string>(StringComparer.Ordinal);
        var cellIndex = 0;

        for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            var row = rows[rowIndex];
            for (var columnIndex = 0; columnIndex < row.Count; columnIndex++)
            {
                var token = row[columnIndex];
                if (!IsCellToken(token))
                {
                    continue;
                }

                var colspan = 1;
                var rowspan = 1;

                if (columnIndex + 1 < row.Count && row[columnIndex + 1] == "lcel")
                {
                    var distance = CheckRight(rows, columnIndex, rowIndex);
                    if (distance > 1)
                    {
                        colspan = Math.Max(colspan, distance);
                    }
                }

                if (rowIndex + 1 < rows.Count && rows[rowIndex + 1][columnIndex] == "ucel")
                {
                    var distance = CheckDown(rows, columnIndex, rowIndex);
                    if (distance > 1)
                    {
                        rowspan = Math.Max(rowspan, distance);
                    }
                }

                if (columnIndex + 1 < row.Count && row[columnIndex + 1] == "xcel")
                {
                    var distanceRight = CheckRight(rows, columnIndex, rowIndex);
                    var distanceDown = CheckDown(rows, columnIndex, rowIndex);

                    var spanValid = true;
                    for (var x = columnIndex; x < columnIndex + distanceRight && spanValid; x++)
                    {
                        for (var y = rowIndex; y < rowIndex + distanceDown; y++)
                        {
                            var key = FormSpanKey(x, y);
                            if (registry2dSpan.Contains(key))
                            {
                                spanValid = false;
                                break;
                            }
                        }
                    }

                    if (spanValid)
                    {
                        for (var x = columnIndex; x < columnIndex + distanceRight; x++)
                        {
                            for (var y = rowIndex; y < rowIndex + distanceDown; y++)
                            {
                                registry2dSpan.Add(FormSpanKey(x, y));
                            }
                        }

                        if (distanceRight > 1)
                        {
                            colspan = Math.Max(colspan, distanceRight);
                        }

                        if (distanceDown > 1)
                        {
                            rowspan = Math.Max(rowspan, distanceDown);
                        }
                    }
                }

                if (colspan > 1 || rowspan > 1)
                {
                    spanMap[cellIndex] = (colspan, rowspan);
                }

                cellIndex += 1;
            }
        }

        return new ReadOnlyDictionary<int, (int Colspan, int Rowspan)>(spanMap);
    }

    private static bool IsSquare(IReadOnlyList<string> sequence)
    {
        var rows = SplitRows(sequence);
        if (rows.Count == 0)
        {
            return true;
        }

        var expectedLength = rows[0].Count + 1;
        foreach (var row in rows)
        {
            if (row.Count + 1 != expectedLength)
            {
                return false;
            }
        }

        return true;
    }

    private static IReadOnlyList<string> PadToSquare(IReadOnlyList<string> sequence, string padTag)
    {
        var rows = SplitRows(sequence);
        if (rows.Count == 0)
        {
            return Array.Empty<string>();
        }

        var maxRowLength = rows.Max(row => row.Count);
        var padded = new List<string>(rows.Count * (maxRowLength + 1));

        foreach (var row in rows)
        {
            var rowCopy = new List<string>(row);
            var padCount = maxRowLength - rowCopy.Count;
            for (var i = 0; i < padCount; i++)
            {
                rowCopy.Add(padTag);
            }

            rowCopy.Add("nl");
            padded.AddRange(rowCopy);
        }

        return padded.AsReadOnly();
    }

    private static List<List<string>> SplitRows(IReadOnlyList<string> sequence)
    {
        var rows = new List<List<string>>();
        var current = new List<string>();

        foreach (var token in sequence)
        {
            if (token == "nl")
            {
                if (current.Count > 0)
                {
                    rows.Add(new List<string>(current));
                    current.Clear();
                }

                continue;
            }

            current.Add(token);
        }

        if (current.Count > 0)
        {
            rows.Add(new List<string>(current));
        }

        return rows;
    }

    private static int CheckDown(IReadOnlyList<List<string>> rows, int x, int y)
    {
        var distance = 1;
        var elem = "ucel";
        var goodList = new HashSet<string>(new[] { "fcel", "ched", "rhed", "srow", "ecel", "lcel", "nl" });

        while (!goodList.Contains(elem) && y < rows.Count - 1)
        {
            y += 1;
            distance += 1;
            elem = rows[y][x];
        }

        if (goodList.Contains(elem))
        {
            distance -= 1;
        }

        return distance;
    }

    private static int CheckRight(IReadOnlyList<List<string>> rows, int x, int y)
    {
        var distance = 1;
        var elem = "lcel";
        var goodList = new HashSet<string>(new[] { "fcel", "ched", "rhed", "srow", "ecel", "ucel", "nl" });

        while (!goodList.Contains(elem) && x < rows[y].Count - 1)
        {
            x += 1;
            distance += 1;
            elem = rows[y][x];
        }

        if (goodList.Contains(elem))
        {
            distance -= 1;
        }

        return distance;
    }

    private static string FormSpanKey(int x, int y) => string.Concat(
        x.ToString(CultureInfo.InvariantCulture),
        "_",
        y.ToString(CultureInfo.InvariantCulture));

    private static bool IsCellToken(string token)
    {
        return token is "fcel" or "ecel" or "xcel" or "ched" or "rhed" or "srow";
    }
}
