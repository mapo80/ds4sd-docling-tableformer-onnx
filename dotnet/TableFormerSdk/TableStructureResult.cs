using SkiaSharp;
using System.Collections.Generic;

namespace TableFormerSdk;

public record TableRegion(float X, float Y, float Width, float Height, string Label);

public sealed class TableStructureResult
{
    public IReadOnlyList<TableRegion> Regions { get; }
    public SKBitmap? OverlayImage { get; }

    public TableStructureResult(IReadOnlyList<TableRegion> regions, SKBitmap? overlay)
    {
        Regions = regions;
        OverlayImage = overlay;
    }
}
