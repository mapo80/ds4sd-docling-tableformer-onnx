# TableFormerTorchSharpSdk

Bootstrap utilities for replicating Docling's TableFormer artifact handling in .NET.

## Features
- Download the canonical TableFormer artifacts for a given variant from Hugging Face.
- Load and validate `tm_config.json` with the same rules used by Docling's Python implementation.
- Canonicalize the configuration JSON and compute its SHA-256 hash for parity checks.
- Compare the .NET canonicalization against the Python baseline hash produced by `scripts/hash_tableformer_config.py`.
- Load the TableFormer word map and safetensors weights, computing per-tensor SHA-256 checksums to guarantee parity with the Python `TFPredictor` bootstrap.
- Reproduce Docling's page resizing and table cropping pipeline with SkiaSharp, yielding byte-identical ROI hashes and summary statistics for every table.
- Convert table crops into normalized `[1, 3, 448, 448]` TorchSharp tensors that match Docling's preprocessing, including per-sample SHA-256 digests and descriptive statistics.
- Execute the TorchSharp neural forward pass and compare logits/coordinates against Docling with 1e-5 tolerance, ensuring identical tag sequences and argmax indices.
- Decode the predicted tag sequences into OTSL/HTML tokens, reproduce Docling's span-aware bounding box corrections, and validate the resulting sequences and bounding boxes against the Python reference with a 1.5e-7 tolerance.
- Rebuild Docling's PDF cell matching stage, translating bounding boxes to page space, reconstructing cell metadata, and comparing against the Python reference with a 2e-5 tolerance on page-space coordinates.
- Mirror Docling's matching post-processing to realign columns and regenerate table cells, verifying final table structures and cell flags against the Python baseline.

## Usage
```csharp
var bootstrapper = new TableFormerArtifactBootstrapper(new DirectoryInfo("artifacts"));
var result = await bootstrapper.EnsureArtifactsAsync();

var reference = await TableFormerConfigReference.LoadAsync(
    Path.Combine("results", "tableformer_config_fast_hash.json"));
result.EnsureConfigMatches(reference);
```

The Python baseline file can be regenerated with:

```bash
PYTHONPATH=tableformer-docling python scripts/hash_tableformer_config.py \
  --variant fast \
  --output results/tableformer_config_fast_hash.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_init_reference.py \
  --variant fast \
  --output results/tableformer_init_fast_reference.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_page_input.py \
  --output results/tableformer_page_input_reference.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_table_crops.py \
  --output results/tableformer_table_crops_reference.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_image_tensors.py \
  --output results/tableformer_image_tensors_reference.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_neural_outputs.py \
  --output results/tableformer_neural_outputs_reference.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_sequence_decoding.py \
  --output results/tableformer_sequence_decoding_reference.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_cell_matching.py \
  --output results/tableformer_cell_matching_reference.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_post_processing.py \
  --output results/tableformer_post_processing_reference.json
```

Once the references are generated, run the targeted tests to validate parity:

```bash
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpInitializationTests.PredictorInitializationMatchesPythonReference
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpPageInputTests.PageInputMatchesPythonReference
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpTableCropTests.TableCroppingMatchesPythonReference
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpImageTensorTests.ImageTensorizationMatchesPythonReference
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpNeuralInferenceTests.NeuralInferenceMatchesPythonReference
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpSequenceDecodingTests.SequenceDecodingMatchesPythonReference
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpCellMatchingTests.CellMatchingMatchesPythonReference
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpPostProcessingTests.PostProcessingMatchesPythonReference
```
