# ds4sd-docling-layout-heron-onnx

## Informazioni generali
Repository di valutazione per il modello `ds4sd/docling-layout-heron`, un detector di layout documentale. L'obiettivo è convertire il modello in diversi formati (ONNX, formato ORT, OpenVINO) e confrontarne le prestazioni su CPU.

## Modello di partenza
Il modello HuggingFace `ds4sd/docling-layout-heron` estrae strutture di layout da pagine di documento, rilevando componenti come paragrafi, titoli, tabelle e figure. Le utilità presenti in questa repo permettono di esportarlo e verificarne la parità rispetto alla versione PyTorch.

## Modelli supportati
- **PyTorch**: modello originale da HuggingFace.
- **ONNX FP32**: versione convertita.
- **ONNX FP32 ottimizzato**: generato con le ottimizzazioni di ONNX Runtime.
- **Formato ORT**: serializzazione del grafo ottimizzato in `.ort` (FP32 e, se supportato, FP16).
- **OpenVINO IR**: modello convertito in formato OpenVINO (FP32).

> La conversione in FP16 è possibile ma l'esecuzione con ONNX Runtime può fallire a causa di operatori non supportati.

## Performance
Benchmark su CPU con input `640×640`, eseguiti in sequenza su due immagini del folder `dataset/` con `--threads-intra 0` e `--threads-inter 1`.

### Tabella di confronto

| Variante                      | Runtime | Precisione | Median (ms) | p95 (ms) | Dimensione (MB) |
|-------------------------------|---------|------------|-------------|----------|-----------------|
| onnx-fp32-cpu                 | Python  | FP32       | 704.90      | 727.85   | 163.53          |
| onnx-fp32-ort                 | Python  | FP32       | 668.98      | 704.83   | 163.97          |
| openvino-fp32-cpu             | Python  | FP32       | 430.03      | 447.35   | 83.07           |
| dotnet-onnx-fp32-cpu          | .NET    | FP32       | 822.09      | 830.62   | 163.53          |
| dotnet-onnx-fp32-ort          | .NET    | FP32       | 926.53      | 935.15   | 163.97          |
| dotnet-openvino-fp32-cpu      | .NET    | FP32       | 576.59      | 594.77   | 83.07           |

I valori provengono dai file `summary.json` e `model_info.json` generati durante il benchmark.

### Dove trovare le misurazioni
Ogni esecuzione salva risultati autoconsistenti in `results/<variant>/run-YYYYMMDD-HHMMSS/` con i seguenti file principali:
- `timings.csv` – latenza per singola immagine.
- `summary.json` – statistiche aggregate (media, mediana, p95).
- `model_info.json` – percorso, dimensione e precisione del modello.
- `env.json`, `config.json`, `manifest.json`, `logs.txt` – contesto di esecuzione e manifest.

## Benchmark su DocLayNet
L'intero set di 20 immagini estratte da `icdar2023-doclaynet.parquet` è stato utilizzato per valutare le tre varianti principali del modello.
OpenVINO risulta il runtime più rapido, con una latenza mediana di ~430 ms e un throughput di oltre 2 immagini al secondo, mantenendo al contempo una dimensione del modello dimezzata rispetto alle versioni ONNX/ORT.
Le varianti ONNX e ORT mostrano prestazioni simili (mediana ~590 ms), ma ORT evidenzia una maggiore variabilità. OpenVINO rileva 3 box in meno rispetto agli altri due, un differenziale di ~0,3 % sul totale delle bounding box.

| Runtime | Model MB | Images | Boxes | Boxes/img | Mean ms | Median ms | P95 ms | Min ms | Max ms | Std ms | Img/s | Boxes/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onnx-fp32-cpu | 163.53 | 20 | 1046 | 52.30 | 593.10 | 590.10 | 656.92 | 522.15 | 752.97 | 64.25 | 1.69 | 88.18 |
| onnx-fp32-ort | 163.97 | 20 | 1046 | 52.30 | 603.14 | 592.92 | 711.99 | 514.61 | 957.76 | 104.70 | 1.66 | 86.71 |
| openvino-fp32-cpu | 83.03 | 20 | 1043 | 52.15 | 492.24 | 430.02 | 713.12 | 392.06 | 1183.23 | 165.96 | 2.03 | 105.94 |

## Conversione e benchmark in ONNX/ORT
1. **Esportazione in ONNX**
   ```bash
   python convert_to_onnx.py --output models/heron-converted.onnx --dataset dataset
   ```
2. **Ottimizzazione**
   ```bash
   python optimize_onnx.py --input models/heron-converted.onnx --output models/heron-optimized.onnx
   ```
3. **Conversione in FP16 (opzionale)**
   ```bash
   python scripts/convert_fp16.py
   ```
4. **Generazione del formato ORT**
   ```bash
   python scripts/convert_to_ort.py
   ```
5. **Benchmark**
   ```bash
   python scripts/bench_python.py --model models/heron-optimized.onnx \
     --images ./dataset --variant-name onnx-fp32-cpu \
     --sequential --threads-intra 0 --threads-inter 1 --target-h 640 --target-w 640

   python scripts/bench_python.py --model models/heron-optimized.ort \
     --images ./dataset --variant-name onnx-fp32-ort \
     --sequential --threads-intra 0 --threads-inter 1 --target-h 640 --target-w 640
   ```

## Conversione e benchmark in OpenVINO
1. **Conversione a IR**
   ```bash
   python scripts/convert_to_openvino.py
   ```
2. **Benchmark**
   ```bash
   python scripts/bench_openvino.py --model models/heron-converted.xml \
     --images ./dataset --variant-name openvino-fp32-cpu \
     --sequential --threads-intra 0 --target-h 640 --target-w 640
   ```

## Requisiti
Installare le dipendenze principali:
```bash
pip install -r requirements.txt
```
I file di modello di grandi dimensioni sono salvati nella cartella `models/` e sono esclusi dal versionamento git.

## SDK .NET 8
La libreria `LayoutSdk` espone un'unica API per eseguire il detector tramite i tre runtime supportati:
- **ONNX Runtime** (file `.onnx`)
- **ONNX Runtime** con grafi ottimizzati in formato `.ort`
- **OpenVINO CSharp API** (modelli IR `.xml`/`.bin`)

### Utilizzo
```csharp
using LayoutSdk;
var sdk = new LayoutSdk(new LayoutSdkOptions(
    "models/heron-optimized.onnx",
    "models/heron-optimized.ort",
    "models/ov-ir/heron-optimized.xml"));
var result = sdk.Process("dataset/gazette_de_france.jpg", overlay: true, LayoutEngine.Openvino);
Console.WriteLine($"Detected: {result.Boxes.Count} elements");
result.OverlayImage?.Encode(SKEncodedImageFormat.Png, 90)
    .SaveTo(File.OpenWrite("overlay.png"));
```
I pacchetti NuGet utilizzati sono le versioni più recenti di `Microsoft.ML.OnnxRuntime`, `OpenVINO.CSharp.API`, `OpenVINO.runtime.ubuntu.24-x86_64` e `SkiaSharp`.
I test xUnit nella cartella `dotnet/LayoutSdk.Tests` verificano la gestione degli errori e la generazione opzionale degli overlay.

### Benchmark .NET
L'applicazione console `LayoutSdk.Benchmarks` replica lo script Python generando le stesse cartelle di output:

```bash
dotnet run --project dotnet/LayoutSdk.Benchmarks/LayoutSdk.Benchmarks.csproj -- \
  --engine Onnx --variant-name dotnet-onnx-fp32-cpu \
  --images dataset --target-h 640 --target-w 640

dotnet run --project dotnet/LayoutSdk.Benchmarks/LayoutSdk.Benchmarks.csproj -- \
  --engine Ort --variant-name dotnet-onnx-fp32-ort \
  --images dataset --target-h 640 --target-w 640
```
I risultati vengono salvati in `results/<variant>/run-YYYYMMDD-HHMMSS/` con gli stessi file `summary.json`, `model_info.json` e relativi manifest.
