namespace TableFormerTorchSharpSdk.Safetensors;

public sealed record SafeTensorTensorRecord(string Name, string Dtype, IReadOnlyList<long> Shape, string Sha256);

public sealed record SafeTensorFileRecord(string FileName, string Sha256, IReadOnlyList<SafeTensorTensorRecord> Tensors);
