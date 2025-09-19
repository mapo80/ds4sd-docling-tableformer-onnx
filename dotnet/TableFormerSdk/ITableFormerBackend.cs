using SkiaSharp;
using System.Collections.Generic;

namespace TableFormerSdk;

public interface ITableFormerBackend
{
    IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath);
}
