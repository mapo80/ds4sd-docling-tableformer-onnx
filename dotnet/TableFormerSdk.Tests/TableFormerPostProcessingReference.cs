using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace TableFormerSdk.Tests;

internal sealed class TableFormerPostProcessingReference
{
    private TableFormerPostProcessingReference(IReadOnlyList<TableFormerPostProcessingSample> samples)
    {
        Samples = samples;
        SamplesByKey = samples.ToDictionary(
            sample => (sample.ImageName, sample.TableIndex),
            sample => sample);
    }

    public IReadOnlyList<TableFormerPostProcessingSample> Samples { get; }

    public IReadOnlyDictionary<(string ImageName, int TableIndex), TableFormerPostProcessingSample> SamplesByKey { get; }

    public static TableFormerPostProcessingReference Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        if (!root.TryGetProperty("samples", out var samplesElement))
        {
            throw new InvalidDataException("Post-processing reference JSON is missing the 'samples' array.");
        }

        var samples = new List<TableFormerPostProcessingSample>();
        foreach (var element in samplesElement.EnumerateArray())
        {
            samples.Add(TableFormerPostProcessingSample.FromJson(element));
        }

        return new TableFormerPostProcessingReference(new ReadOnlyCollection<TableFormerPostProcessingSample>(samples));
    }
}

internal sealed class TableFormerPostProcessingSample
{
    private TableFormerPostProcessingSample(
        string imageName,
        int tableIndex,
        TableFormerPostProcessingSection input,
        TableFormerPostProcessingSection output,
        IReadOnlyList<TableFormerDocOutputReference> docOutput,
        string inputTableCellsSha,
        string inputMatchesSha,
        string outputTableCellsSha,
        string outputMatchesSha,
        string docOutputSha)
    {
        ImageName = imageName;
        TableIndex = tableIndex;
        Input = input;
        Output = output;
        DocOutput = docOutput;
        InputTableCellsSha256 = inputTableCellsSha;
        InputMatchesSha256 = inputMatchesSha;
        OutputTableCellsSha256 = outputTableCellsSha;
        OutputMatchesSha256 = outputMatchesSha;
        DocOutputSha256 = docOutputSha;
    }

    public string ImageName { get; }

    public int TableIndex { get; }

    public TableFormerPostProcessingSection Input { get; }

    public TableFormerPostProcessingSection Output { get; }

    public IReadOnlyList<TableFormerDocOutputReference> DocOutput { get; }

    public string InputTableCellsSha256 { get; }

    public string InputMatchesSha256 { get; }

    public string OutputTableCellsSha256 { get; }

    public string OutputMatchesSha256 { get; }

    public string DocOutputSha256 { get; }

    public static TableFormerPostProcessingSample FromJson(JsonElement element)
    {
        var imageName = element.GetProperty("image_name").GetString() ?? string.Empty;
        var tableIndex = element.GetProperty("table_index").GetInt32();
        var input = TableFormerPostProcessingSection.FromJson(element.GetProperty("input"));
        var output = TableFormerPostProcessingSection.FromJson(element.GetProperty("output"));

        var docOutputElement = element.GetProperty("doc_output");
        var docOutput = new List<TableFormerDocOutputReference>();
        foreach (var entry in docOutputElement.EnumerateArray())
        {
            docOutput.Add(TableFormerDocOutputReference.FromJson(entry));
        }

        return new TableFormerPostProcessingSample(
            imageName,
            tableIndex,
            input,
            output,
            new ReadOnlyCollection<TableFormerDocOutputReference>(docOutput),
            element.GetProperty("input_table_cells_sha256").GetString() ?? string.Empty,
            element.GetProperty("input_matches_sha256").GetString() ?? string.Empty,
            element.GetProperty("output_table_cells_sha256").GetString() ?? string.Empty,
            element.GetProperty("output_matches_sha256").GetString() ?? string.Empty,
            element.GetProperty("doc_output_sha256").GetString() ?? string.Empty);
    }
}

internal sealed class TableFormerPostProcessingSection
{
    private TableFormerPostProcessingSection(
        IReadOnlyList<TableFormerCellReference> tableCells,
        string tableCellsSha,
        IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> matches,
        string matchesSha,
        IReadOnlyList<TableFormerPdfCellReference> pdfCells,
        int maxCellId)
    {
        TableCells = tableCells;
        TableCellsSha256 = tableCellsSha;
        Matches = matches;
        MatchesSha256 = matchesSha;
        PdfCells = pdfCells;
        MaxCellId = maxCellId;
    }

    public IReadOnlyList<TableFormerCellReference> TableCells { get; }

    public string TableCellsSha256 { get; }

    public IReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>> Matches { get; }

    public string MatchesSha256 { get; }

    public IReadOnlyList<TableFormerPdfCellReference> PdfCells { get; }

    public int MaxCellId { get; }

    public static TableFormerPostProcessingSection FromJson(JsonElement element)
    {
        var tableCells = new List<TableFormerCellReference>();
        foreach (var cellElement in element.GetProperty("table_cells").EnumerateArray())
        {
            var bbox = cellElement
                .GetProperty("bbox")
                .EnumerateArray()
                .Select(v => v.GetDouble())
                .ToArray();

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

            tableCells.Add(new TableFormerCellReference(
                cellElement.GetProperty("cell_id").GetInt32(),
                cellElement.GetProperty("row_id").GetInt32(),
                cellElement.GetProperty("column_id").GetInt32(),
                new ReadOnlyCollection<double>(bbox),
                cellElement.GetProperty("cell_class").GetInt32(),
                cellElement.GetProperty("label").GetString() ?? string.Empty,
                cellElement.TryGetProperty("multicol_tag", out var multicolElement) ? multicolElement.GetString() : null,
                colspan,
                rowspan,
                colspanValue,
                rowspanValue));
        }

        var matches = new Dictionary<string, IReadOnlyList<TableFormerMatchReference>>(StringComparer.Ordinal);
        foreach (var matchProperty in element.GetProperty("matches").EnumerateObject())
        {
            var matchList = new List<TableFormerMatchReference>();
            foreach (var matchElement in matchProperty.Value.EnumerateArray())
            {
                matchList.Add(new TableFormerMatchReference(
                    matchElement.GetProperty("table_cell_id").GetInt32(),
                    matchElement.TryGetProperty("iopdf", out var iopdfElement) ? iopdfElement.GetDouble() : (double?)null,
                    matchElement.TryGetProperty("iou", out var iouElement) ? iouElement.GetDouble() : (double?)null,
                    matchElement.TryGetProperty("post", out var postElement) ? postElement.GetDouble() : (double?)null));
            }

            matches[matchProperty.Name] = new ReadOnlyCollection<TableFormerMatchReference>(matchList);
        }

        var pdfCells = new List<TableFormerPdfCellReference>();
        foreach (var pdfElement in element.GetProperty("pdf_cells").EnumerateArray())
        {
            pdfCells.Add(new TableFormerPdfCellReference(
                pdfElement.GetProperty("id").GetString() ?? string.Empty,
                pdfElement.TryGetProperty("text", out var textElement) ? textElement.GetString() ?? string.Empty : string.Empty,
                new ReadOnlyCollection<double>(pdfElement.GetProperty("bbox").EnumerateArray().Select(v => v.GetDouble()).ToArray())));
        }

        var tableCellsSha = element.TryGetProperty("table_cells_sha256", out var tcShaElement)
            ? tcShaElement.GetString() ?? string.Empty
            : string.Empty;
        var matchesSha = element.TryGetProperty("matches_sha256", out var matchesShaElement)
            ? matchesShaElement.GetString() ?? string.Empty
            : string.Empty;
        var maxCellId = element.TryGetProperty("max_cell_id", out var maxElement)
            ? maxElement.GetInt32()
            : -1;

        return new TableFormerPostProcessingSection(
            new ReadOnlyCollection<TableFormerCellReference>(tableCells),
            tableCellsSha,
            new ReadOnlyDictionary<string, IReadOnlyList<TableFormerMatchReference>>(matches),
            matchesSha,
            new ReadOnlyCollection<TableFormerPdfCellReference>(pdfCells),
            maxCellId);
    }
}

internal sealed class TableFormerDocOutputReference
{
    internal TableFormerDocOutputReference(
        string pdfCellId,
        int tableCellId,
        IReadOnlyList<double> bbox,
        int columnSpan,
        int rowSpan,
        bool columnHeader,
        bool rowHeader,
        bool rowSection)
    {
        PdfCellId = pdfCellId;
        TableCellId = tableCellId;
        BoundingBox = bbox;
        ColumnSpan = columnSpan;
        RowSpan = rowSpan;
        ColumnHeader = columnHeader;
        RowHeader = rowHeader;
        RowSection = rowSection;
    }

    public string PdfCellId { get; }

    public int TableCellId { get; }

    public IReadOnlyList<double> BoundingBox { get; }

    public int ColumnSpan { get; }

    public int RowSpan { get; }

    public bool ColumnHeader { get; }

    public bool RowHeader { get; }

    public bool RowSection { get; }

    public static TableFormerDocOutputReference FromJson(JsonElement element)
    {
        return new TableFormerDocOutputReference(
            element.GetProperty("pdf_cell_id").GetString() ?? string.Empty,
            element.GetProperty("table_cell_id").GetInt32(),
            new ReadOnlyCollection<double>(element.GetProperty("bbox").EnumerateArray().Select(v => v.GetDouble()).ToArray()),
            element.GetProperty("col_span").GetInt32(),
            element.GetProperty("row_span").GetInt32(),
            element.GetProperty("column_header").GetBoolean(),
            element.GetProperty("row_header").GetBoolean(),
            element.GetProperty("row_section").GetBoolean());
    }
}
