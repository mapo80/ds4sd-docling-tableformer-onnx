# ds4sd-docling-tableformer-onnx

## Informazioni generali
Repository di implementazione .NET per i modelli TableFormer di Docling. Il progetto fornisce:
- Implementazione .NET con TorchSharp della pipeline completa di TableFormer
- Script Python di riferimento per validazione e benchmarking
- Test di parità tra implementazione Python e .NET

**Nota ONNX:** Sono state tentate conversioni dei modelli in formato ONNX per ottimizzare le performance, ma non hanno avuto successo. L'implementazione attuale utilizza TorchSharp con i modelli originali in formato safetensors.

## Modelli supportati
Gli artifact HuggingFace disponibili in [`ds4sd/docling-models`](https://huggingface.co/ds4sd/docling-models/tree/main/model_artifacts/tableformer) includono:
- **TableFormer fast** – ottimizzata per la latenza
- **TableFormer accurate** – ottimizzata per la qualità dell'estrazione

## SDK TorchSharp (.NET 9)

### Requisiti
- .NET 9 SDK
- Dataset FinTabNet per benchmark e validazione (opzionale, vedere sezione "Preparazione dataset")

### Download automatico dei modelli
La libreria `TableFormerTorchSharpSdk` **scarica automaticamente** i modelli dalla release GitHub [`v1.0.0`](https://github.com/mapo80/ds4sd-docling-tableformer-onnx/releases/tag/v1.0.0). Non è necessario scaricare manualmente i file.

Al primo avvio, il bootstrapper:
1. Scarica l'archivio `tableformer-{variant}.zip` dalla release (`fast` o `accurate`)
2. Verifica l'hash SHA-256 dell'archivio
3. Estrae la configurazione `tm_config.json` e i pesi `tableformer_{variant}.safetensors`
4. Salva gli artifact nella directory `artifacts/` con la seguente struttura:
   ```
   artifacts/
   └── model_artifacts/
       └── tableformer/
           └── fast/                          # o "accurate"
               ├── tm_config.json             # Configurazione del modello (contiene il WORDMAP inline)
               └── tableformer_fast.safetensors  # Pesi del modello
   ```
5. Calcola hash SHA-256 per verifica di integrità

**Release GitHub:** `mapo80/ds4sd-docling-tableformer-onnx` tag `v1.0.0`  
**Varianti disponibili:** `fast`, `accurate`

**File estratti automaticamente:**
- `tm_config.json` – configurazione del modello (include `dataset_wordmap` inline)
- `tableformer_{variant}.safetensors` – pesi del modello (es. `tableformer_fast.safetensors`)

### Installazione e utilizzo

#### 1. Clonare il repository
```bash
git clone <repository-url>
cd ds4sd-docling-tableformer-onnx
```

#### 2. Generare i riferimenti Python (opzionale, per test di parità)
I test di parità richiedono file di riferimento generati dalla pipeline Python:

```bash
# Installare dipendenze Python
pip install -r requirements.txt

# Generare tutti i riferimenti necessari
PYTHONPATH=tableformer-docling python scripts/hash_tableformer_config.py \
  --variant fast --output results/tableformer_config_fast_hash.json

PYTHONPATH=tableformer-docling python scripts/export_tableformer_init_reference.py \
  --variant fast --output results/tableformer_init_fast_reference.json

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

#### 3. Eseguire i test di parità (opzionale)
```bash
# Test completo della suite TorchSharp
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharp

# Oppure test specifici per ogni fase della pipeline:
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpInitializationTests
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpPageInputTests
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpTableCropTests
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpImageTensorTests
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpNeuralInferenceTests
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpSequenceDecodingTests
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpCellMatchingTests
dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpPostProcessingTests
```

#### 4. Utilizzare l'SDK nel codice

**Esempio completo: pipeline end-to-end**

```csharp
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.Model;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Results;
using TableFormerTorchSharpSdk.Tensorization;

class Program
{
    static async Task Main(string[] args)
    {
        // 1. Bootstrap: scarica automaticamente i modelli dalla GitHub release (fast/accurate)
        var artifactsRoot = new DirectoryInfo("artifacts");
        using var bootstrapper = new TableFormerArtifactBootstrapper(
            artifactsRoot,
            TableFormerModelVariant.Fast);  // oppure TableFormerModelVariant.Accurate

        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();
        Console.WriteLine($"Modello scaricato: {bootstrapResult.ModelDirectory.FullName}");

        // 2. Inizializza il predictor (carica pesi safetensors)
        var initSnapshot = await bootstrapResult.InitializePredictorAsync();
        Console.WriteLine($"Tensori caricati: {initSnapshot.TensorDigests.Count}");

        // 3. Crea il modello neurale
        TableFormerNeuralModel.ConfigureThreading(Environment.ProcessorCount);
        using var neuralModel = new TableFormerNeuralModel(
            bootstrapResult.ConfigSnapshot,
            initSnapshot,
            bootstrapResult.ModelDirectory);

        // 4. Prepara i componenti della pipeline
        var decoder = new TableFormerSequenceDecoder(initSnapshot);
        var cellMatcher = new TableFormerCellMatcher(bootstrapResult.ConfigSnapshot);
        var cropper = new TableFormerTableCropper();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
        var preparer = new TableFormerPageInputPreparer();
        var postProcessor = new TableFormerMatchingPostProcessor();
        var assembler = new TableFormerDoclingResponseAssembler();

        // 5. Processa un'immagine
        var imageFile = new FileInfo("path/to/table-image.png");
        var decodedImage = TableFormerDecodedPageImage.Decode(imageFile);

        // 6. Pipeline completa
        var pageSnapshot = preparer.PreparePageInput(decodedImage);
        var cropSnapshot = cropper.PrepareTableCrops(decodedImage, pageSnapshot.TableBoundingBoxes);

        foreach (var crop in cropSnapshot.TableCrops)
        {
            // Tensorizzazione
            using var tensorSnapshot = tensorizer.CreateTensor(crop);

            // Inferenza neurale
            using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);

            // Decodifica sequenza
            var decoded = decoder.Decode(prediction);

            // Cell matching
            var matchingResult = cellMatcher.MatchCells(pageSnapshot, crop, decoded);
            var matchingDetails = matchingResult.ToMatchingDetails();

            // Post-processing
            var processed = postProcessor.Process(
                matchingDetails.ToMutable(),
                correctOverlappingCells: false);

            // Assembla risultato finale
            var result = assembler.Assemble(processed, decoded, sortRowColIndexes: true);

            // Stampa risultati
            Console.WriteLine($"Tabella rilevata: {result.Data.Rows.Count} righe, {result.Data.Cols.Count} colonne");
            foreach (var cell in result.TableCells.Where(c => !c.IsProjected))
            {
                Console.WriteLine($"  Cella [{cell.RowIndex}, {cell.ColIndex}]: bbox={cell.Bbox}");
            }
        }
    }
}
```

**Esempio semplice: solo bootstrap e inizializzazione**

```csharp
using System;
using System.IO;
using System.Threading.Tasks;
using TableFormerTorchSharpSdk.Artifacts;

// Download automatico modelli (eseguito solo la prima volta)
var artifactsRoot = new DirectoryInfo("artifacts");
using var bootstrapper = new TableFormerArtifactBootstrapper(
    artifactsRoot,
    TableFormerModelVariant.Fast);

// Scarica dalla release GitHub se non già presente
var bootstrapResult = await bootstrapper.EnsureArtifactsAsync();

// Inizializza e verifica i pesi
var predictorSnapshot = await bootstrapResult.InitializePredictorAsync();

Console.WriteLine($"Modello pronto!");
Console.WriteLine($"Directory: {bootstrapResult.ModelDirectory.FullName}");
Console.WriteLine($"Tensori verificati: {predictorSnapshot.TensorDigests.Count}");
Console.WriteLine($"Word map: {predictorSnapshot.WordMap.Count} tokens");
```

### Verifica rapida via CLI
Per verificare che entrambe le varianti (`fast`, `accurate`) producano gli stessi risultati dei riferimenti Docling è disponibile la CLI:

```bash
dotnet run --project dotnet/TableFormerTorchSharpSdk.Cli/TableFormerTorchSharpSdk.Cli.csproj
```

Il comando scarica (se necessario) i pesi dalla release GitHub, esegue la pipeline sulle immagini in `dataset/FinTabNet/benchmark` e confronta l'output con i file di riferimento in `results/`.  
Per rigenerare i riferimenti (ad esempio dopo aver aggiornato gli artifact della release) usare:

```bash
dotnet run --project dotnet/TableFormerTorchSharpSdk.Cli/TableFormerTorchSharpSdk.Cli.csproj -- --variant accurate --update-reference
dotnet run --project dotnet/TableFormerTorchSharpSdk.Cli/TableFormerTorchSharpSdk.Cli.csproj -- --variant fast --update-reference
```

### Struttura directory
```
ds4sd-docling-tableformer-onnx/
├── artifacts/                                 # Modelli scaricati automaticamente
│   └── model_artifacts/
│       └── tableformer/
│           └── fast/                          # o "accurate"
│               ├── tm_config.json
│               └── tableformer_fast.safetensors
├── dataset/                                   # Dataset per benchmark (opzionale)
│   └── FinTabNet/
│       └── benchmark/                         # Immagini PNG
├── dotnet/
│   ├── TableFormerTorchSharpSdk/              # Libreria principale
│   ├── TableFormerSdk.Tests/                  # Test di parità
│   ├── TableFormerTorchSharpSdk.Benchmarks/   # CLI per performance
│   └── TableFormerVisualizer/                 # Tool di visualizzazione
├── results/                                   # File di riferimento Python
├── scripts/                                   # Script Python per riferimenti
└── tableformer-docling/                       # Implementazione Python di riferimento
```

### Preparazione dataset (opzionale)
Il dataset FinTabNet è richiesto solo per eseguire i benchmark di performance. Posizionare le immagini PNG in:
```
dataset/FinTabNet/benchmark/
```

### Benchmark delle performance
```bash
dotnet run -c Release \
  --project dotnet/TableFormerTorchSharpSdk.Benchmarks \
  -- --dataset dataset/FinTabNet/benchmark \
     --baseline results/perf_baseline.json \
     --iterations 3 \
     --label optimized
```

Opzioni disponibili:
- `--dataset <path>`: directory con le immagini PNG del benchmark
- `--runs-dir <path>`: directory output per i JSON delle iterazioni (default: `results/perf_runs`)
- `--report <path>`: percorso del report Markdown (default: `results/performance_report.md`)
- `--baseline <path>`: baseline di riferimento per il confronto
- `--iterations <n>`: numero di iterazioni da eseguire
- `--label <name>`: etichetta per identificare il run
- `--num-threads <n>`: numero di thread per TorchSharp
- `--skip-reference-check`: salta la verifica delle predizioni contro il riferimento Python

Ogni iterazione genera:
- Timings per documento
- Breakdown per fase della pipeline
- Metadata (configurazione thread TorchSharp, versione .NET, commit git, ecc.)

## Componenti della pipeline .NET
La libreria `TableFormerTorchSharpSdk` implementa tutte le fasi della pipeline Docling:

1. **Bootstrap artifact** – Download e validazione modelli dalla release GitHub
2. **Page preparation** – Resize e normalizzazione delle pagine con SkiaSharp
3. **Table cropping** – Estrazione delle regioni contenenti tabelle
4. **Tensorization** – Conversione immagini in tensori normalizzati [1,3,448,448]
5. **Neural inference** – Forward pass con TorchSharp (encoder + decoder con attention)
6. **Sequence decoding** – Decodifica tag OTSL/HTML e coordinate bounding box
7. **Cell matching** – Matching celle tra predizioni neurali e contenuto PDF
8. **Post-processing** – Riallineamento colonne e ricostruzione struttura tabellare
9. **Response assembly** – Generazione output finale in formato Docling

Ogni fase garantisce parità numerica con Python attraverso test automatizzati con tolleranze specifiche (1e-5 per logits, 1.5e-7 per bounding box, 2e-5 per coordinate pagina).

## Script Python disponibili

### Benchmark Python
```bash
python scripts/benchmark_docling_python.py  # Benchmark con pipeline Docling completa
```

### Validazione e confronto
```bash
python compare_kpi.py                       # Confronto KPI tra modelli
python compare_tableformer_results.py       # Confronto risultati TableFormer
```

### Export riferimenti per test di parità
```bash
# Hash configurazione
python scripts/hash_tableformer_config.py --variant fast

# Export snapshot per ogni fase della pipeline
python scripts/export_tableformer_init_reference.py
python scripts/export_tableformer_page_input.py
python scripts/export_tableformer_table_crops.py
python scripts/export_tableformer_image_tensors.py
python scripts/export_tableformer_neural_outputs.py
python scripts/export_tableformer_sequence_decoding.py
python scripts/export_tableformer_cell_matching.py
python scripts/export_tableformer_post_processing.py
```
