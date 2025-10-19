using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;

using TableFormerTorchSharpSdk.Utilities;

namespace TableFormerSdk.Tests;

internal sealed class TableFormerDoclingReference
{
    private TableFormerDoclingReference(
        IReadOnlyDictionary<string, TableFormerDoclingReferencePage> pages,
        IReadOnlyDictionary<(string ImageName, int TableIndex), string> tablesByKey)
    {
        Pages = pages;
        TablesByKey = tablesByKey;
    }

    public IReadOnlyDictionary<string, TableFormerDoclingReferencePage> Pages { get; }

    public IReadOnlyDictionary<(string ImageName, int TableIndex), string> TablesByKey { get; }

    public static TableFormerDoclingReference Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);

        var pages = new Dictionary<string, TableFormerDoclingReferencePage>(StringComparer.Ordinal);
        var tables = new Dictionary<(string ImageName, int TableIndex), string>();

        foreach (var property in document.RootElement.EnumerateObject())
        {
            var imageName = property.Name;
            var value = property.Value;
            var numTables = value.GetProperty("num_tables").GetInt32();
            var tablesElement = value.GetProperty("tables");

            var canonicalTables = new List<string>(tablesElement.GetArrayLength());
            var tableIndex = 0;
            foreach (var tableElement in tablesElement.EnumerateArray())
            {
                var canonical = JsonCanonicalizer.GetCanonicalJson(tableElement);
                canonicalTables.Add(canonical);
                tables[(imageName, tableIndex)] = canonical;
                tableIndex += 1;
            }

            pages[imageName] = new TableFormerDoclingReferencePage(imageName, numTables, canonicalTables);
        }

        return new TableFormerDoclingReference(
            new ReadOnlyDictionary<string, TableFormerDoclingReferencePage>(pages),
            new ReadOnlyDictionary<(string ImageName, int TableIndex), string>(tables));
    }
}

internal sealed class TableFormerDoclingReferencePage
{
    public TableFormerDoclingReferencePage(
        string imageName,
        int numTables,
        IReadOnlyList<string> canonicalTables)
    {
        ImageName = imageName;
        NumTables = numTables;
        Tables = canonicalTables;
    }

    public string ImageName { get; }

    public int NumTables { get; }

    public IReadOnlyList<string> Tables { get; }
}
