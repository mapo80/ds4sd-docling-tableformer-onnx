using SkiaSharp;
using System.Collections.Generic;

namespace LayoutSdk;

internal interface ILayoutBackend
{
    IReadOnlyList<BoundingBox> Infer(SKBitmap image);
}
