using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;

using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Decoding;

namespace TableFormerTorchSharpSdk.Results;

public sealed class TableFormerDoclingBoundingBox
{
    public TableFormerDoclingBoundingBox(double bottom, double left, double right, double top, string? token)
    {
        Bottom = bottom;
        Left = left;
        Right = right;
        Top = top;
        Token = token ?? string.Empty;
    }

    public double Bottom { get; }

    public double Left { get; }

    public double Right { get; }

    public double Top { get; }

    public string Token { get; }

    public Dictionary<string, object?> ToDictionary()
    {
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["b"] = Bottom,
            ["l"] = Left,
            ["r"] = Right,
            ["t"] = Top,
            ["token"] = Token,
        };
    }

    public static TableFormerDoclingBoundingBox FromArray(IReadOnlyList<double> values, string? token)
    {
        if (values.Count != 4)
        {
            throw new ArgumentException("Bounding box array must contain exactly four elements.", nameof(values));
        }

        return new TableFormerDoclingBoundingBox(values[3], values[0], values[2], values[1], token);
    }

    public static TableFormerDoclingBoundingBox FromBoundingBox(TableFormerBoundingBox bbox, string? token)
    {
        return new TableFormerDoclingBoundingBox(bbox.Bottom, bbox.Left, bbox.Right, bbox.Top, token);
    }
}

public sealed class TableFormerDoclingCellResponse
{
    private readonly List<TableFormerDoclingBoundingBox> _textCellBoundingBoxes;

    public TableFormerDoclingCellResponse(int cellId)
    {
        CellId = cellId;
        RowSpan = 1;
        ColSpan = 1;
        StartRowOffsetIndex = -1;
        EndRowOffsetIndex = -1;
        StartColOffsetIndex = -1;
        EndColOffsetIndex = -1;
        IndentationLevel = 0;
        _textCellBoundingBoxes = new List<TableFormerDoclingBoundingBox>();
    }

    public int CellId { get; }

    public TableFormerDoclingBoundingBox? BoundingBox { get; set; }

    public int RowSpan { get; set; }

    public int ColSpan { get; set; }

    public int StartRowOffsetIndex { get; set; }

    public int EndRowOffsetIndex { get; set; }

    public int StartColOffsetIndex { get; set; }

    public int EndColOffsetIndex { get; set; }

    public int IndentationLevel { get; }

    public bool ColumnHeader { get; set; }

    public bool RowHeader { get; set; }

    public bool RowSection { get; set; }

    public IReadOnlyList<TableFormerDoclingBoundingBox> TextCellBoundingBoxes =>
        new ReadOnlyCollection<TableFormerDoclingBoundingBox>(_textCellBoundingBoxes.ToArray());

    public void AddTextCellBoundingBox(TableFormerDoclingBoundingBox boundingBox)
    {
        ArgumentNullException.ThrowIfNull(boundingBox);
        _textCellBoundingBoxes.Add(boundingBox);
    }

    public TableFormerDoclingCellResponse CloneWithoutTextCells()
    {
        return new TableFormerDoclingCellResponse(CellId)
        {
            BoundingBox = BoundingBox,
            RowSpan = RowSpan,
            ColSpan = ColSpan,
            StartRowOffsetIndex = StartRowOffsetIndex,
            EndRowOffsetIndex = EndRowOffsetIndex,
            StartColOffsetIndex = StartColOffsetIndex,
            EndColOffsetIndex = EndColOffsetIndex,
            ColumnHeader = ColumnHeader,
            RowHeader = RowHeader,
            RowSection = RowSection,
        };
    }

    public Dictionary<string, object?> ToDictionary()
    {
        var dictionary = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["cell_id"] = CellId,
            ["bbox"] = BoundingBox?.ToDictionary() ?? new Dictionary<string, object?>(StringComparer.Ordinal),
            ["row_span"] = RowSpan,
            ["col_span"] = ColSpan,
            ["start_row_offset_idx"] = StartRowOffsetIndex,
            ["end_row_offset_idx"] = EndRowOffsetIndex,
            ["start_col_offset_idx"] = StartColOffsetIndex,
            ["end_col_offset_idx"] = EndColOffsetIndex,
            ["indentation_level"] = IndentationLevel,
            ["text_cell_bboxes"] = _textCellBoundingBoxes
                .Select(box => box.ToDictionary())
                .ToList(),
            ["column_header"] = ColumnHeader,
            ["row_header"] = RowHeader,
            ["row_section"] = RowSection,
        };

        return dictionary;
    }
}

public sealed class TableFormerDoclingPrediction
{
    public TableFormerDoclingPrediction(
        IReadOnlyList<TableFormerNormalizedBoundingBox> boundingBoxes,
        IReadOnlyList<int> classPredictions,
        IReadOnlyList<int> tagSequence,
        IReadOnlyList<string> rsSequence,
        IReadOnlyList<string> htmlSequence)
    {
        BoundingBoxes = boundingBoxes ?? Array.Empty<TableFormerNormalizedBoundingBox>();
        ClassPredictions = classPredictions ?? Array.Empty<int>();
        TagSequence = tagSequence ?? Array.Empty<int>();
        RsSequence = rsSequence ?? Array.Empty<string>();
        HtmlSequence = htmlSequence ?? Array.Empty<string>();
    }

    public IReadOnlyList<TableFormerNormalizedBoundingBox> BoundingBoxes { get; }

    public IReadOnlyList<int> ClassPredictions { get; }

    public IReadOnlyList<int> TagSequence { get; }

    public IReadOnlyList<string> RsSequence { get; }

    public IReadOnlyList<string> HtmlSequence { get; }

    public Dictionary<string, object?> ToDictionary()
    {
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["bboxes"] = BoundingBoxes.Select(box => box.ToArray()).ToList(),
            ["classes"] = ClassPredictions.ToList(),
            ["tag_seq"] = TagSequence.ToList(),
            ["rs_seq"] = RsSequence.ToList(),
            ["html_seq"] = HtmlSequence.ToList(),
        };
    }
}

public sealed class TableFormerDoclingPredictDetails
{
    public TableFormerDoclingPredictDetails(
        double iouThreshold,
        TableFormerBoundingBox tableBoundingBox,
        IReadOnlyList<TableFormerBoundingBox> predictionBoundingBoxesPage,
        TableFormerDoclingPrediction prediction,
        IReadOnlyList<TableFormerMutablePdfCell> pdfCells,
        int pageWidth,
        int pageHeight,
        IReadOnlyList<TableFormerMutableTableCell> tableCells,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>> matches,
        IReadOnlyList<TableFormerDoclingCellResponse> doclingResponses,
        int numCols,
        int numRows)
    {
        IouThreshold = iouThreshold;
        TableBoundingBox = tableBoundingBox;
        PredictionBoundingBoxesPage = new ReadOnlyCollection<TableFormerBoundingBox>(
            (predictionBoundingBoxesPage ?? Array.Empty<TableFormerBoundingBox>()).ToArray());
        Prediction = prediction;
        PdfCells = new ReadOnlyCollection<TableFormerMutablePdfCell>((pdfCells ?? Array.Empty<TableFormerMutablePdfCell>()).ToArray());
        PageWidth = pageWidth;
        PageHeight = pageHeight;
        TableCells = new ReadOnlyCollection<TableFormerMutableTableCell>((tableCells ?? Array.Empty<TableFormerMutableTableCell>()).ToArray());
        Matches = matches ?? new ReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>>(new Dictionary<string, IReadOnlyList<TableFormerMutableMatch>>(StringComparer.Ordinal));
        DoclingResponses = new ReadOnlyCollection<TableFormerDoclingCellResponse>((doclingResponses ?? Array.Empty<TableFormerDoclingCellResponse>()).ToArray());
        NumCols = numCols;
        NumRows = numRows;
    }

    public double IouThreshold { get; }

    public TableFormerBoundingBox TableBoundingBox { get; }

    public IReadOnlyList<TableFormerBoundingBox> PredictionBoundingBoxesPage { get; }

    public TableFormerDoclingPrediction Prediction { get; }

    public IReadOnlyList<TableFormerMutablePdfCell> PdfCells { get; }

    public int PageWidth { get; }

    public int PageHeight { get; }

    public IReadOnlyList<TableFormerMutableTableCell> TableCells { get; }

    public IReadOnlyDictionary<string, IReadOnlyList<TableFormerMutableMatch>> Matches { get; }

    public IReadOnlyList<TableFormerDoclingCellResponse> DoclingResponses { get; }

    public int NumCols { get; }

    public int NumRows { get; }

    public Dictionary<string, object?> ToDictionary()
    {
        var matchesDictionary = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var pair in Matches)
        {
            matchesDictionary[pair.Key] = pair.Value.Select(match => match.ToDictionary()).ToList();
        }

        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["iou_threshold"] = IouThreshold,
            ["table_bbox"] = TableBoundingBox.ToArray(),
            ["prediction_bboxes_page"] = PredictionBoundingBoxesPage.Select(box => box.ToArray()).ToList(),
            ["prediction"] = Prediction.ToDictionary(),
            ["pdf_cells"] = PdfCells.Select(cell => cell.ToDictionary()).ToList(),
            ["page_width"] = Convert.ToDouble(PageWidth, CultureInfo.InvariantCulture),
            ["page_height"] = Convert.ToDouble(PageHeight, CultureInfo.InvariantCulture),
            ["table_cells"] = TableCells.Select(cell => cell.ToDictionary()).ToList(),
            ["matches"] = matchesDictionary,
            ["docling_responses"] = DoclingResponses.Select(response => response.ToDictionary()).ToList(),
            ["num_cols"] = NumCols,
            ["num_rows"] = NumRows,
        };
    }
}

public sealed class TableFormerDoclingTablePrediction
{
    public TableFormerDoclingTablePrediction(
        IReadOnlyList<TableFormerDoclingCellResponse> tfResponses,
        TableFormerDoclingPredictDetails predictDetails)
    {
        TfResponses = new ReadOnlyCollection<TableFormerDoclingCellResponse>(tfResponses.ToArray());
        PredictDetails = predictDetails ?? throw new ArgumentNullException(nameof(predictDetails));
    }

    public IReadOnlyList<TableFormerDoclingCellResponse> TfResponses { get; }

    public TableFormerDoclingPredictDetails PredictDetails { get; }

    public Dictionary<string, object?> ToDictionary()
    {
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["tf_responses"] = TfResponses.Select(response => response.ToDictionary()).ToList(),
            ["predict_details"] = PredictDetails.ToDictionary(),
        };
    }
}
