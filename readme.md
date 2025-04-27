# Systémová príručka

## Celková charakteristika architektúry

Systém slúži na analýzu a vizualizáciu objektov detegovaných v snímkach miestností s cieľom identifikovať a zakresliť pozície objektov v 3D priestore. Využíva YOLO model na rozpoznávanie objektov a umožňuje interaktívne anotácie. Výsledky sú ukladané do cache pre rýchlejší opätovný prístup.

### Základné komponenty

- **YOLO (YOLOv8)**  
YOLOv8 model slúži na detekciu objektov v snímkach. Použité triedy sú definované a filtrované podľa špecifických kritérií (napr. "bowl", "cup", "bottle").

- **Cache management**  
Ukladá a načítava anotácie objektov, čím optimalizuje opakované spracovanie.

- **Image management**  
Spravuje sady obrázkov, spracovanie a analýzu obrázkov podľa konkrétnych adresárov.

- **3D Visualizer (Matplotlib)**  
Zabezpečuje vizualizáciu výsledkov v 3D priestore, vrátane zakreslenia miestnosti a objektov.

### Štruktúra súborov

- **datasets** (`ds_1`, `ds_2`, `ds_3`, `ds_4`)  
Obsahujú snímky miestností pre jednotlivé experimenty.

- **Cache folder (`_cache`)**  
Uchováva anotácie v JSON formáte.

- **Debug folder (`_debug`)**  
Uchováva obrázky s zakreslenými detekciami pre debugging.

### Komponenty

#### YOLO Model

Používa `yolov8n.pt` pre detekciu objektov.

#### Funkcie spracovania

- **parse_filename**: Parsovanie názvu súboru na získanie polohy kamery a uhlov.
- **pixel_to_angles**: Konverzia pixelových koordinátov na horizontálne a vertikálne uhly.
- **bounding_box_to_line**: Prevod informácií o bounding boxoch na 3D vektory smerov.

### Proces spracovania

- Načítanie obrázkov zo špecifikovaných adresárov.
- Detekcia objektov pomocou YOLO.
- Interaktívne anotácie (voliteľné).
- Ukladanie a načítanie výsledkov z cache.
- Vizualizácia objektov v 3D priestore.

### Práca s manuálnymi anotáciami

Manuálne anotácie sa využívajú pri ladení a debugovaní systému. Ak je parameter `interactive_annotation` nastavený na `True`, používateľ môže manuálne pridávať anotácie. Tieto anotácie sa ukladajú do cache vo formáte JSON, čo umožňuje ich opakované použitie pri ďalšom spracovaní bez nutnosti manuálneho opakovania anotácie.

### Nastavenie systému

#### Požiadavky

- Python 3.x
- YOLOv8
- OpenCV
- Matplotlib
- Numpy

#### Štruktúra adresárov

```
project/
├── ds_1/
├── ds_2/
├── ds_3/
├── ds_4/
├── yolov8n.pt
├── dtc_v6.py
```

#### Štruktúra snímok
`x_y_z_angleToXAxis_angleToZAxis.jpg`
`x` - Koordinát x
`y` - Koordinát y
`z` - Koordinát z
`angleToXAxis` - uhol otočenia kamery voči uhlu X
`angleToZAxis` - uhol otočenia kamery voči uhlu Z

#### Spustenie skriptu

Spustenie analýzy a vizualizácie:

```bash
python yolo-slam.py
```

### Konfigurácia systému

Hlavné konfiguračné parametre (`yolo-slam.py`):

```python
logs = False
interactive_annotation = True
use_cached_annotations = True

room_dimensions = (170, 120, 100)
horizontal_fov = 90
vertical_fov = 60
```

### Cache manažment

Cache súbory sú uložené vo formáte JSON v príslušnom adresári `_cache` pre rýchlejšie načítanie anotácií.

### Debugovanie

Výsledky detekcie sú vizualizované a uložené v adresári `_debug`. Pri zapnutom debug móde (nastavenie `logs = True`) systém poskytuje detailné výpisy o priebehu spracovania, ktoré sú užitočné pri ladení.

### Doplňujúce informácie

- Triedy objektov môžu byť premenované pre konzistentnosť (napr. „wine glass“ na „cup“).
- Používateľ môže vytvárať nové debug a cache adresáre podľa potreby, pričom systém automaticky skontroluje ich existenciu.
- Funkcie systému využívajú geometrické transformácie pre presné zakreslenie detegovaných objektov v priestore.

### Záverečné poznámky

Tento manuál poskytuje jasný prehľad o architektúre systému, potrebných komponentoch, nastavení, spôsobe spustenia systému a detailné informácie k práci s manuálnymi anotáciami, debugovaniu a ďalším komponentom potrebným pre správne fungovanie aplikácie.

