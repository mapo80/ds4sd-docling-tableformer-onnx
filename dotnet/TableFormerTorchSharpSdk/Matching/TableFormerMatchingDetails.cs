using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;

using TableFormerTorchSharpSdk.PagePreparation;

namespace TableFormerTorchSharpSdk.Matching;

public sealed class TableFormerMatchingDetails
{
    public TableFormerMatchingDetails(
        double iouThreshold,
        TableFormerBoundingBox tableBoundingBox,
        IReadOnlyList<TableFormerBoundingBox> predictionBoundingBoxes,
        IReadOnlyList<TableFormerMutableTableCell> tableCells,
        IReadOnlyList<TableFormerMutablePdfCell> pdfCells,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>> matches,
        int pageWidth,
        int pageHeight)
    {
        IouThreshold = iouThreshold;
        TableBoundingBox = tableBoundingBox;
        PredictionBoundingBoxesPage = new ReadOnlyCollection<TableFormerBoundingBox>(predictionBoundingBoxes.ToArray());
        TableCells = new ReadOnlyCollection<TableFormerMutableTableCell>(tableCells.Select(cell => cell.Clone()).ToArray());
        PdfCells = new ReadOnlyCollection<TableFormerMutablePdfCell>(pdfCells.Select(cell => cell.Clone()).ToArray());
        Matches = new ReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>>(
            matches.ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<TableFormerMutableMatch>)new ReadOnlyCollection<TableFormerMutableMatch>(
                    pair.Value.Select(match => match.Clone()).ToArray()),
                StringComparer.Ordinal));
        PageWidth = pageWidth;
        PageHeight = pageHeight;
    }

    private TableFormerMatchingDetails(
        double iouThreshold,
        TableFormerBoundingBox tableBoundingBox,
        IList<TableFormerBoundingBox> predictionBoundingBoxes,
        IList<TableFormerMutableTableCell> tableCells,
        IList<TableFormerMutablePdfCell> pdfCells,
        IDictionary<string, IList<TableFormerMutableMatch>> matches,
        int pageWidth,
        int pageHeight,
        bool cloneInternal)
    {
        IouThreshold = iouThreshold;
        TableBoundingBox = tableBoundingBox;
        PredictionBoundingBoxesPage = new ReadOnlyCollection<TableFormerBoundingBox>(predictionBoundingBoxes);
        TableCells = new ReadOnlyCollection<TableFormerMutableTableCell>(tableCells);
        PdfCells = new ReadOnlyCollection<TableFormerMutablePdfCell>(pdfCells);
        Matches = new ReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>>(
            matches.ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<TableFormerMutableMatch>)new ReadOnlyCollection<TableFormerMutableMatch>(pair.Value),
                StringComparer.Ordinal));
        PageWidth = pageWidth;
        PageHeight = pageHeight;
    }

    public double IouThreshold { get; }

    public TableFormerBoundingBox TableBoundingBox { get; }

    public IReadOnlyList<TableFormerBoundingBox> PredictionBoundingBoxesPage { get; }

    public IReadOnlyList<TableFormerMutableTableCell> TableCells { get; }

    public IReadOnlyList<TableFormerMutablePdfCell> PdfCells { get; }

    public IReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>> Matches { get; }

    public int PageWidth { get; }

    public int PageHeight { get; }

    public TableFormerMatchingDetails Clone() =>
        new(
            IouThreshold,
            TableBoundingBox,
            PredictionBoundingBoxesPage,
            TableCells,
            PdfCells,
            Matches,
            PageWidth,
            PageHeight);

    public MutableTableFormerMatchingDetails ToMutable()
    {
        return new MutableTableFormerMatchingDetails(
            IouThreshold,
            TableBoundingBox,
            PredictionBoundingBoxesPage.Select(b => b).ToList(),
            TableCells.Select(cell => cell.Clone()).ToList(),
            PdfCells.Select(cell => cell.Clone()).ToList(),
            Matches.ToDictionary(
                pair => pair.Key,
                pair => (IList<TableFormerMutableMatch>)pair.Value.Select(match => match.Clone()).ToList(),
                StringComparer.Ordinal),
            PageWidth,
            PageHeight);
    }

    public static TableFormerMatchingDetails FromMutable(MutableTableFormerMatchingDetails mutable)
    {
        ArgumentNullException.ThrowIfNull(mutable);
        return new TableFormerMatchingDetails(
            mutable.IouThreshold,
            mutable.TableBoundingBox,
            mutable.PredictionBoundingBoxesPage,
            mutable.TableCells,
            mutable.PdfCells,
            mutable.Matches,
            mutable.PageWidth,
            mutable.PageHeight,
            cloneInternal: true);
    }
}

public sealed class MutableTableFormerMatchingDetails
{
    public MutableTableFormerMatchingDetails(
        double iouThreshold,
        TableFormerBoundingBox tableBoundingBox,
        IList<TableFormerBoundingBox> predictionBoundingBoxes,
        IList<TableFormerMutableTableCell> tableCells,
        IList<TableFormerMutablePdfCell> pdfCells,
        IDictionary<string, IList<TableFormerMutableMatch>> matches,
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

    public double IouThreshold { get; set; }

    public TableFormerBoundingBox TableBoundingBox { get; set; }

    public IList<TableFormerBoundingBox> PredictionBoundingBoxesPage { get; }

    public IList<TableFormerMutableTableCell> TableCells { get; }

    public IList<TableFormerMutablePdfCell> PdfCells { get; }

    public IDictionary<string, IList<TableFormerMutableMatch>> Matches { get; }

    public int PageWidth { get; set; }

    public int PageHeight { get; set; }
}

public sealed class TableFormerMutablePdfCell
{
    public TableFormerMutablePdfCell(string id, TableFormerBoundingBox boundingBox, string text)
    {
        Id = id;
        BoundingBox = boundingBox;
        Text = text;
    }

    public string Id { get; set; }

    public TableFormerBoundingBox BoundingBox { get; set; }

    public string Text { get; set; }

    public TableFormerMutablePdfCell Clone() => new(Id, BoundingBox, Text);

    public Dictionary<string, object?> ToDictionary()
    {
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["id"] = Id,
            ["bbox"] = BoundingBox.ToArray(),
            ["text"] = Text,
        };
    }
}

public sealed class TableFormerMutableMatch
{
    public TableFormerMutableMatch(int tableCellId, double? intersectionOverPdf, double? intersectionOverUnion, double? post)
    {
        TableCellId = tableCellId;
        IntersectionOverPdf = intersectionOverPdf;
        IntersectionOverUnion = intersectionOverUnion;
        PostScore = post;
    }

    public int TableCellId { get; set; }

    public double? IntersectionOverPdf { get; set; }

    public double? IntersectionOverUnion { get; set; }

    public double? PostScore { get; set; }

    public TableFormerMutableMatch Clone() => new(TableCellId, IntersectionOverPdf, IntersectionOverUnion, PostScore);

    public Dictionary<string, object?> ToDictionary()
    {
        var dict = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["table_cell_id"] = TableCellId,
        };

        if (IntersectionOverPdf.HasValue)
        {
            dict["iopdf"] = IntersectionOverPdf.Value;
        }

        if (IntersectionOverUnion.HasValue)
        {
            dict["iou"] = IntersectionOverUnion.Value;
        }

        if (PostScore.HasValue)
        {
            dict["post"] = PostScore.Value;
        }

        return dict;
    }

    public static TableFormerMutableMatch FromDictionary(IDictionary<string, object?> dictionary)
    {
        ArgumentNullException.ThrowIfNull(dictionary);
        if (!dictionary.TryGetValue("table_cell_id", out var tableCellIdValue) || tableCellIdValue is null)
        {
            throw new InvalidOperationException("Match dictionary missing 'table_cell_id'.");
        }

        var tableCellId = Convert.ToInt32(tableCellIdValue, CultureInfo.InvariantCulture);
        double? iopdf = null;
        double? iou = null;
        double? post = null;

        if (dictionary.TryGetValue("iopdf", out var iopdfValue) && iopdfValue is not null)
        {
            iopdf = Convert.ToDouble(iopdfValue, CultureInfo.InvariantCulture);
        }

        if (dictionary.TryGetValue("iou", out var iouValue) && iouValue is not null)
        {
            iou = Convert.ToDouble(iouValue, CultureInfo.InvariantCulture);
        }

        if (dictionary.TryGetValue("post", out var postValue) && postValue is not null)
        {
            post = Convert.ToDouble(postValue, CultureInfo.InvariantCulture);
        }

        return new TableFormerMutableMatch(tableCellId, iopdf, iou, post);
    }
}

public sealed class TableFormerMutableTableCell
{
    private readonly Dictionary<string, object?> _data;

    public TableFormerMutableTableCell(
        int cellId,
        int rowId,
        int columnId,
        IReadOnlyList<double> boundingBox,
        int cellClass,
        string label,
        string multicolTag,
        int? colspan,
        int? rowspan)
    {
        _data = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["cell_id"] = cellId,
            ["row_id"] = rowId,
            ["column_id"] = columnId,
            ["bbox"] = boundingBox.ToArray(),
            ["cell_class"] = cellClass,
            ["label"] = label,
            ["multicol_tag"] = multicolTag ?? string.Empty,
        };

        if (colspan.HasValue)
        {
            _data["colspan"] = colspan.Value;
        }

        if (rowspan.HasValue)
        {
            _data["rowspan"] = rowspan.Value;
        }
    }

    private TableFormerMutableTableCell(Dictionary<string, object?> data)
    {
        _data = new Dictionary<string, object?>(data, StringComparer.Ordinal);
    }

    public Dictionary<string, object?> Data => _data;

    public int CellId
    {
        get => GetInt("cell_id");
        set => _data["cell_id"] = value;
    }

    public int RowId
    {
        get => GetInt("row_id");
        set => _data["row_id"] = value;
    }

    public int ColumnId
    {
        get => GetInt("column_id");
        set => _data["column_id"] = value;
    }

    public double[] BoundingBox
    {
        get => GetBoundingBox();
        set => _data["bbox"] = value;
    }

    public int CellClass
    {
        get => GetInt("cell_class");
        set => _data["cell_class"] = value;
    }

    public string Label
    {
        get => GetString("label");
        set => _data["label"] = value;
    }

    public string MulticolTag
    {
        get => GetString("multicol_tag");
        set => _data["multicol_tag"] = value;
    }

    public int? Colspan
    {
        get => GetNullableInt("colspan");
        set
        {
            if (value.HasValue)
            {
                _data["colspan"] = value.Value;
            }
            else
            {
                _data.Remove("colspan");
            }
        }
    }

    public int? Rowspan
    {
        get => GetNullableInt("rowspan");
        set
        {
            if (value.HasValue)
            {
                _data["rowspan"] = value.Value;
            }
            else
            {
                _data.Remove("rowspan");
            }
        }
    }

    public int? ColspanValue
    {
        get => GetNullableInt("colspan_val");
        set
        {
            if (value.HasValue)
            {
                _data["colspan_val"] = value.Value;
            }
            else
            {
                _data.Remove("colspan_val");
            }
        }
    }

    public int? RowspanValue
    {
        get => GetNullableInt("rowspan_val");
        set
        {
            if (value.HasValue)
            {
                _data["rowspan_val"] = value.Value;
            }
            else
            {
                _data.Remove("rowspan_val");
            }
        }
    }

    public double? ColumnScore
    {
        get => GetNullableDouble("column_score");
        set
        {
            if (value.HasValue)
            {
                _data["column_score"] = value.Value;
            }
            else
            {
                _data.Remove("column_score");
            }
        }
    }

    public double? RowScore
    {
        get => GetNullableDouble("row_score");
        set
        {
            if (value.HasValue)
            {
                _data["row_score"] = value.Value;
            }
            else
            {
                _data.Remove("row_score");
            }
        }
    }

    public TableFormerMutableTableCell Clone() => new(_data);

    public Dictionary<string, object?> ToDictionary() => new(_data, StringComparer.Ordinal);

    public static TableFormerMutableTableCell FromDictionary(IDictionary<string, object?> dictionary)
    {
        ArgumentNullException.ThrowIfNull(dictionary);
        return new TableFormerMutableTableCell(new Dictionary<string, object?>(dictionary, StringComparer.Ordinal));
    }

    private int GetInt(string key)
    {
        if (!_data.TryGetValue(key, out var value) || value is null)
        {
            throw new InvalidOperationException($"Table cell missing required '{key}'.");
        }

        return Convert.ToInt32(value, CultureInfo.InvariantCulture);
    }

    private int? GetNullableInt(string key)
    {
        if (!_data.TryGetValue(key, out var value) || value is null)
        {
            return null;
        }

        return Convert.ToInt32(value, CultureInfo.InvariantCulture);
    }

    private double[] GetBoundingBox()
    {
        if (!_data.TryGetValue("bbox", out var value) || value is null)
        {
            throw new InvalidOperationException("Table cell missing 'bbox'.");
        }

        if (value is double[] array)
        {
            return array;
        }

        if (value is IReadOnlyList<double> list)
        {
            return list.ToArray();
        }

        if (value is IEnumerable<double> enumerable)
        {
            return enumerable.ToArray();
        }

        throw new InvalidOperationException("Table cell 'bbox' is not a numeric sequence.");
    }

    private string GetString(string key)
    {
        if (!_data.TryGetValue(key, out var value) || value is null)
        {
            return string.Empty;
        }

        return Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
    }

    private double? GetNullableDouble(string key)
    {
        if (!_data.TryGetValue(key, out var value) || value is null)
        {
            return null;
        }

        return Convert.ToDouble(value, CultureInfo.InvariantCulture);
    }
}
