namespace TableFormerTorchSharpSdk.Assets;

/// <summary>
/// Identifies the GitHub release containing TableFormer assets.
/// </summary>
public sealed record TableFormerGithubReleaseOptions(
    string Repository = "mapo80/ds4sd-docling-tableformer-onnx",
    string Tag = "v1.0.0")
{
    /// <summary>
    /// Compute the base download URL for the configured release.
    /// </summary>
    public string BuildBaseUrl() => $"https://github.com/{Repository}/releases/download/{Tag}";
}
