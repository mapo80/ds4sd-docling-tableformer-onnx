# TableFormer Docling Extraction

Questa cartella contiene una versione minimale dei moduli Python necessari per
eseguire inferenza con TableFormer così come implementato in Docling, senza
installare l'intero pacchetto `docling`.

## Contenuto

- `docling_ibm_models/`: copia dei moduli TableFormer originali
  distribuiti con Docling.
- `tableformer_docling/`: piccoli wrapper per scaricare gli artefatti del
  modello e lanciare l'inferenza.

## Download dei pesi

I pesi e i file di configurazione vengono scaricati automaticamente dalla
repository HuggingFace `ds4sd/docling-models` (sottocartella
`model_artifacts/tableformer/fast`) al primo utilizzo. Gli artefatti vengono
salvati nella sottocartella `artifacts/`.

## Come usare l'inferenza Python

Esempio minimo:

```python
from pathlib import Path
import numpy as np
from PIL import Image

from tableformer_docling.predictor import TableFormerDocling

runner = TableFormerDocling(artifacts_dir=Path("tableformer-docling/artifacts"))
image = np.asarray(Image.open("path/to/table_image.png").convert("RGB"))
results = runner.predict_page(image)
```

Il metodo `predict_page` restituisce la stessa struttura di dati prodotta da
Docling (`tf_responses`, `predict_details`, ecc.).

Per esempi completi vedere `run_tableformer_docling.py` nella root del
repository.
