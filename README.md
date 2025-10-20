# ds4sd-docling-tableformer-onnx

## Informazioni generali
Repository di valutazione per i modelli `ds4sd/docling-models` dedicati al riconoscimento di tabelle TableFormer.
L'obiettivo è convertire le varianti *fast* e *accurate* in diversi formati (ONNX, formato ORT, OpenVINO), verificarne la parità rispetto ai modelli originali e misurarne le prestazioni su CPU.

## Modelli di partenza
Gli artifact HuggingFace disponibili in [`ds4sd/docling-models`](https://huggingface.co/ds4sd/docling-models/tree/main/model_artifacts/tableformer) contengono due configurazioni principali:

- **TableFormer fast** – ottimizzata per la latenza
- **TableFormer accurate** – ottimizzata per la qualità dell'estrazione

Entrambe condividono la stessa pipeline di preprocessing (resize a 448×448, normalizzazione RGB) ma differiscono per dimensioni e numero di layer.

## Formati supportati
- **PyTorch**: modelli originali in formato `safetensors`
- **ONNX FP32**: esportazione del grafo encoder/decoder TableFormer
- **ONNX ottimizzato (ORT)**: grafo compatto per l'esecuzione con ONNX Runtime
- **OpenVINO IR**: modelli convertiti in `.xml`/`.bin` per CPU Intel

> La conversione in FP16 e INT8 è in corso di validazione; le istruzioni verranno aggiornate una volta completata la parità.

## Pipeline di conversione (bozza)
1. **Esportazione in ONNX**
   ```bash
   python convert_to_onnx.py --model-name tableformer-fast --output models/tableformer-fast.onnx --dataset dataset
   python convert_to_onnx.py --model-name tableformer-accurate --output models/tableformer-accurate.onnx --dataset dataset
   ```
2. **Ottimizzazione**
   ```bash
   python optimize_onnx.py --input models/tableformer-fast.onnx --output models/tableformer-fast-optimized.onnx
   python optimize_onnx.py --input models/tableformer-accurate.onnx --output models/tableformer-accurate-optimized.onnx
   ```
3. **Formato ORT**
   ```bash
   python scripts/convert_to_ort.py --input models/tableformer-fast-optimized.onnx --output models/tableformer-fast.ort
   python scripts/convert_to_ort.py --input models/tableformer-accurate-optimized.onnx --output models/tableformer-accurate.ort
   ```
4. **Conversione in OpenVINO**
   ```bash
   python scripts/convert_to_openvino.py --input models/tableformer-fast.onnx --output-dir models/ov-ir-fast
   python scripts/convert_to_openvino.py --input models/tableformer-accurate.onnx --output-dir models/ov-ir-accurate
   ```

## Benchmark
Gli script di benchmark generano cartelle `results/<variant>/run-YYYYMMDD-HHMMSS/` con:
- `timings.csv` – latenza per immagine
- `summary.json` – statistiche aggregate (media, mediana, p95)
- `model_info.json` – percorso, dimensione, precisione del modello
- `env.json`, `config.json`, `manifest.json`, `logs.txt` – contesto di esecuzione

Esempio di benchmark Python per la variante fast:
```bash
python scripts/bench_python.py --model models/tableformer-fast-optimized.onnx \
  --images ./dataset --variant-name onnx-fast-fp32-cpu \
  --sequential --threads-intra 0 --threads-inter 1 --target-h 448 --target-w 448
```

## SDK TorchSharp (.NET 9)
La libreria `TableFormerTorchSharpSdk` replica in .NET la pipeline di Docling basata su TorchSharp, fornendo strumenti per scaricare gli artifact Hugging Face, convalidare le configurazioni e garantire la parità numerica con il bootstrap Python.

### Componenti principali
- **Bootstrap artifact** – `TableFormerArtifactBootstrapper` scarica `tm_config.json`, word map e pesi `safetensors`, producendo snapshot firmati per verificare hash e file scaricati.
- **Inizializzazione del predictor** – `TableFormerPredictorInitializer` ricostruisce la pipeline TorchSharp eseguendo controlli di parità sui tensori rispetto al riferimento Python.
- **Pipeline completa** – i moduli in `PagePreparation`, `Tensorization`, `Model`, `Decoding`, `Matching` e `Results` riproducono il resize delle pagine, la generazione dei tensori, il forward pass e la ricostruzione delle celle con le stesse tolleranze impiegate da Docling.

### Utilizzo rapido
```csharp
using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Configuration;

var artifactsRoot = new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, "artifacts"));
using var bootstrapper = new TableFormerArtifactBootstrapper(artifactsRoot, variant: "fast");

var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
bootstrapResult.EnsureConfigMatches(await TableFormerConfigReference.LoadAsync(
    "results/tableformer_config_fast_hash.json"));

var predictorSnapshot = await bootstrapResult.InitializePredictorAsync();
Console.WriteLine($"Tensors verified: {predictorSnapshot.TensorDigests.Count}");
```

### Test e benchmark .NET
- **Test unitari**: `dotnet test TableFormerSdk.sln --filter TableFormerTorchSharp` per eseguire i test di parità TorchSharp.
- **Benchmark**: `dotnet/TableFormerTorchSharpSdk.Benchmarks` riproduce gli script Python salvando tempi, overlay e metadati.

```bash
dotnet run --project dotnet/TableFormerTorchSharpSdk.Benchmarks/TableFormerTorchSharpSdk.Benchmarks.csproj -- \
  --engine TorchSharp --variant-name Fast \
  --artifacts-root artifacts --output results/benchmarks \
  --runs-per-image 5 --warmup 1 --target-h 448 --target-w 448
```

## Requisiti
Installare le dipendenze principali:
```bash
pip install -r requirements.txt
```
I modelli di grandi dimensioni vengono salvati nella cartella `models/` ed esclusi dal versionamento git.
