
---

# Podstawy Sztucznej Inteligencji

## Raport z projektu

<p align="right">Korzeniak Jakub, Graboś Mateusz</p>


# 1. Określenie tematu i celu projektu, analiza wymagań

Tematem naszego projektu jest zbadanie sposobów w jaki sztuczna inteligencja może "słyszeć" muzykę, oraz ją interpretować.
 Celem naszego projektu jest klasyfikacja utworów muzycznych używając algorytmów sztucznej inteligencji, oraz danych zawierających wiele różnych parametrów tych utworów.

# 2. Zbiór Danych i ich przygotowanie

  

## Zbiór Danych

  

Użyliśmy  następującego  zbioru  danych:  https://www.kaggle.com/datasets/purumalgi/music-genre-classification

Zbiór zawiera zarówno dane uczące jak i dane testowe, jest ich wystarczająco dużo aby przeprowadzić na nich uczenie maszynowe.
  

## Czyszczenie danych: `cleandata.py`

### 🔍 Opis

Skrypt  `cleandata.py`  służy  do  czyszczenia  oraz  wstępnego  przetwarzania  danych  z  plików  test.csv  i  train.csv:

-  Usuwa  utwory  z  brakującymi  wyrazami  w  jakiejkolwiek  kolumnie

-  Usuwa  utwory  ze  tylułami  lub  autorami  ze  znakami  spoza  ASCII

  

Następnie  zapisuje  zmienione  pliki  pod  nazwami  clean_test.csv  oraz  clean_train.csv.

  

### 💻 Kod

#### Ważniejsze fragmenty kodu:

Wczytanie  danych  i  usuwanie  niepkompletnych  utworów

```python

test_df  =  pd.read_csv(data_path_test)

test_df.dropna(inplace=True)

```

Znalezienie  i  usunięcie  utworów  z  tylułami  lub  autorami  ze  znakami  spoza  ASCII

```python

def  is_ascii(s):

return  bool(re.match(r'^[\x00-\x7F]+$',  str(s)))

clean_test_df  =  test_df[test_df["Artist  Name"].apply(is_ascii)  &  test_df["Track  Name"].apply(is_ascii)]

```

Te  same  działania  następnie  czyszczą  plik  train.csv.

W obydwu oczyszczonych plikach, pozostawiono nazwy wykonawców oraz tytuły piosenek w celu lepszej czytelności. Kolumny te nie będą interpretowane przez model AI.

# 3. Wybór i implementacja modelu AI

### 📖 Opis

W celu stworzenia modelu wybrano odpowiednie kolumny z danymi z pliku clean_train.csv stworzonego w poprzednim kamieniu milowym.
Odrzucono takie kolumny jak: "Artist name", "Track name", "Popularity".
Następnie przeskalowano dane wejściowe tak aby miały odchylenie standardowe równe jeden, a średnią równą zero, co ma zapewnić równy wkład każdego parametru do modelu.

W kolejnym kroku wydzielono dane wyjściowe, czyli kolumnę "Class" w której przechowywany jest numer oznaczający gatunek muzyczny, są to odpowiednio:

Acoustic/Folk - 0, Alt_Music - 1, Blues - 2, Bollywood - 3, Country - 4, HipHop - 5,Indie Alt - 6, Instrumental - 7,Metal - 8, Pop - 9, Rock - 10

Następnie zbudowano prosty model z trzech warstw sieci neuronowych, przesłano do niego wydzielone wcześniej dane i rozpoczęto jego trening.

### 🔣 Kod

Budowa modelu przebiega następująco

```python

model = Sequential([
    Dense(64, activation='relu',),
    Dense(64, activation='relu'),
    Dense(11, activation='softmax')  
])

```

Trenowanie modelu wygląda następująco

```python

model.fit(X_train, y_train, epochs=50,
          validation_data=(X_test, y_test),
          class_weight=class_weight_dict,
          callbacks=[early_stop]
          )

```