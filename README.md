---
# Podstawy Sztucznej Inteligencji
## Raport z projektu
<p align="right">Korzeniak Jakub, Graboś Mateusz</p>

# Pierwszy kamień milowy
## czyszczenie danych: `cleandata.py`

### 🔍 Opis
Skrypt `cleandata.py` służy do czyszczenia oraz wstępnego przetwarzania danych z plików test.csv i train.csv:
- Usuwa utwory z brakującymi wyrazami w jakiejkolwiek kolumnie
- Usuwa utwory ze tylułami lub autorami ze znakami spoza ASCII

Następnie zapisuje zmienione pliki pod nazwami clean_test.csv oraz clean_train.csv.

### 💻 Kod
Ważniejsze fragmenty kodu:
-wczytanie danych i usuwanie niepkompletnych utworów
```python
test_df = pd.read_csv(data_path_test)


test_df.dropna(inplace=True)

```
-Znalezienie i usunięcie utworów z tylułami lub autorami ze znakami spoza ASCII
```python
def is_ascii(s):
    return bool(re.match(r'^[\x00-\x7F]+$', str(s)))


clean_test_df = test_df[test_df["Artist Name"].apply(is_ascii) & test_df["Track Name"].apply(is_ascii)]
```
Te same działania następnie czyszczą plik train.csv.



