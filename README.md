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

## SDK .NET 8
La libreria `TableFormerSdk` espone API per eseguire i modelli TableFormer tramite i runtime supportati:
- **ONNX Runtime** (`.onnx`)
- **ONNX Runtime** con grafi ottimizzati (`.ort`)
- **OpenVINO CSharp API** (`.xml`/`.bin`)

### Selezione della variante
`TableFormerModelVariant` permette di scegliere l'artefact *Fast* o *Accurate* al momento dell'inferenza. I percorsi vengono forniti tramite `TableFormerModelPaths`.

### Esempio d'uso
```csharp
using TableFormerSdk;

var sdk = new TableFormerSdk(new TableFormerSdkOptions(
    new TableFormerModelPaths(
        fastModelPath: "models/tableformer-fast.onnx",
        accurateModelPath: "models/tableformer-accurate.onnx")));

var result = sdk.Process(
    imagePath: "dataset/gazette_de_france.jpg",
    overlay: true,
    runtime: TableFormerRuntime.Onnx,
    variant: TableFormerModelVariant.Fast);

Console.WriteLine($"Detected regions: {result.Regions.Count}");
result.OverlayImage?.Encode(SKEncodedImageFormat.Png, 90)
    .SaveTo(File.OpenWrite("overlay.png"));
```
I pacchetti NuGet utilizzati includono `Microsoft.ML.OnnxRuntime`, `OpenVINO.CSharp.API`, `OpenVINO.runtime.ubuntu.24-x86_64` e `SkiaSharp`.

### Report di inferenza FinTabNet (.NET)
Per validare l'integrazione dello SDK, il progetto console `TableFormerSdk.Samples` elabora tre immagini del dataset FinTabNet sfruttando le annotazioni del parquet come backend fittizio (in attesa dei modelli ONNX reali). Il comando da eseguire è:

```bash
dotnet run --project dotnet/TableFormerSdk.Samples/TableFormerSdk.Samples.csproj
```

Il programma salva overlay e dati tabellari in `results/tableformer-net-sample/`. La tabella seguente riassume il numero di regioni individuate e le metriche geometriche estratte dallo SDK:

| Immagine | Regioni | Larghezza media (px) | Altezza media (px) | Area media (px²) | Overlay |
| --- | ---: | ---: | ---: | ---: | --- |
| HAL.2015.page_43.pdf_125177.png | 128 | 27,6 | 10,0 | 275,9 | [PNG](results/tableformer-net-sample/HAL.2015.page_43.pdf_125177_overlay.png) |
| HAL.2009.page_77.pdf_125051.png | 30 | 39,9 | 10,0 | 399,0 | [PNG](results/tableformer-net-sample/HAL.2009.page_77.pdf_125051_overlay.png) |
| HAL.2017.page_79.pdf_125247.png | 9 | 46,3 | 10,0 | 463,3 | [PNG](results/tableformer-net-sample/HAL.2017.page_79.pdf_125247_overlay.png) |

Per ogni immagine viene prodotto anche un file `*_regions.json` con le coordinate di tutte le celle estratte, utile per confronti futuri con l'inferenza ONNX reale.

#### Confronto varianti fast/accurate (ONNX Runtime vs OpenVINO)
Lo stesso campione FinTabNet viene eseguito con **sei inferenze consecutive** per ogni combinazione di runtime (.NET ONNX Runtime/OpenVINO) e variante del modello (fast/accurate), scartando la prima misura di warm-up e salvando le restanti cinque latenze. La tabella riporta la media aritmetica delle cinque misurazioni utili:

| Immagine | ONNX fast (ms) | ONNX accurate (ms) | OpenVINO fast (ms) | OpenVINO accurate (ms) | Vincitore |
| --- | ---: | ---: | ---: | ---: | --- |
| HAL.2015.page_43.pdf_125177.png | 1,56 | 1,07 | 0,99 | 1,08 | OpenVINO fast |
| HAL.2009.page_77.pdf_125051.png | 0,42 | 0,28 | 0,48 | 0,39 | ONNX accurate |
| HAL.2017.page_79.pdf_125247.png | 0,21 | 0,17 | 0,16 | 0,15 | OpenVINO accurate |

Le 60 misurazioni valide mostrano una media complessiva di **0,73 ms** per ONNX fast, **0,51 ms** per ONNX accurate, **0,54 ms** per OpenVINO fast e **0,54 ms** per OpenVINO accurate, con quest'ultima leggermente dietro alla variante ONNX accurate che risulta la più rapida sul campione fittizio. I dettagli integrali (min/max e singole misurazioni) sono disponibili in `results/tableformer-net-sample/perf_comparison.json`.

### Test e benchmark .NET
- **Test unitari**: `dotnet test TableFormerSdk.sln`
- **Benchmark**: l'applicazione console `TableFormerSdk.Benchmarks` replica gli script Python e salva gli stessi artifact.

```bash
dotnet run --project dotnet/TableFormerSdk.Benchmarks/TableFormerSdk.Benchmarks.csproj -- \
  --engine Onnx --variant-name dotnet-fast-onnx-fp32-cpu \
  --images dataset --target-h 448 --target-w 448
```

## Requisiti
Installare le dipendenze principali:
```bash
pip install -r requirements.txt
```
I modelli di grandi dimensioni vengono salvati nella cartella `models/` ed esclusi dal versionamento git.
