using SkiaSharp;
using System.Collections.Generic;

namespace LayoutSdk;

public record BoundingBox(float X, float Y, float Width, float Height, string Label);

public class LayoutResult
{
    public IReadOnlyList<BoundingBox> Boxes { get; }
    public SKBitmap? OverlayImage { get; }

    public LayoutResult(IReadOnlyList<BoundingBox> boxes, SKBitmap? overlay)
    {
        Boxes = boxes;
        OverlayImage = overlay;
    }
}
