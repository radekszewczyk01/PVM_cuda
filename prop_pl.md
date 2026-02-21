# Rozwój percepcji neuromorficznej: Skalowanie modelu Predictive Vision Model za pomocą CUDA C i frameworka Lava dla procesora Loihi 2

Ewolucja sztucznej inteligencji była w dużej mierze definiowana przez dwie odrębne epoki: początkową erę opartą na regułach, skupioną na klasycznej logice, oraz obecną, drugą generację zdominowaną przez detekcję i percepcję poprzez głębokie uczenie (deep learning). Chociaż głębokie konwolucyjne sieci neuronowe (CNN) odniosły niezwykły sukces w analizie klatek wideo, pozostają one fundamentalnie "kruche", opierając się na deterministycznych widokach zdarzeń, którym brakuje zrozumienia kontekstu i zdroworozsądkowego wnioskowania.1 W miarę jak branża zmierza w kierunku trzeciej generacji AI, charakteryzującej się autonomiczną adaptacją i poznaniem zbliżonym do ludzkiego, ograniczenia tradycyjnych, opartych na klatkach, synchronicznych architektur obliczeniowych stają się coraz bardziej widoczne. Predictive Vision Model (PVM) stanowi pionierską próbę wypełnienia tej luki, wykorzystując w pełni rekurencyjną, nienadzorowaną architekturę, która uczy się dynamiki rzeczywistości poprzez ciągłą predykcję czasową.2

Obecny projekt ma na celu przekroczenie historycznych ograniczeń wydajnościowych modelu PVM poprzez przeprojektowanie jego podłoża treningowego dla środowisk obliczeniowych o wysokiej wydajności (HPC) oraz zmapowanie jego silnika wnioskowania na procesor neuromorficzny Intel Loihi 2. Poprzez przejście z obecnej implementacji, opartej w dużej mierze na wrapperach w Pythonie, na zoptymalizowany język CUDA C i framework Lava, badania te odpowiadają na krytyczną potrzebę energooszczędnego śledzenia obiektów o niskich opóźnieniach w systemach autonomicznych.5 Inicjatywa ta wykorzystuje nieodłączne zalety wizji opartej na zdarzeniach (event-based vision) – takie jak wysoki zakres dynamiki (HDR) i mikrosekundowa rozdzielczość czasowa – jednocześnie wykorzystując lokalne reguły uczenia PVM, aby ominąć problemy zanikającego gradientu, które nękają głębokie sieci jednokierunkowe (feedforward).2

## Ramy teoretyczne i Predictive Vision Model

Predictive Vision Model opiera się na neurologicznej zasadzie, że mózg jest w istocie maszyną predykcyjną. Zamiast jedynie klasyfikować statyczne migawki, PVM próbuje zrozumieć równania ruchu wartości sensorycznych poprzez asocjację obecnych wejść z ich przyszłymi stanami.2 Podejście to jest realizowane za pośrednictwem rozproszonej sieci jednostek pamięci asocjacyjnej, z których każda funkcjonuje jako koder predykcyjny z wbudowanym wąskim gardłem kompresji (compression bottleneck).2 To wąskie gardło zmusza jednostki do przewidywania przyszłości w oparciu tylko o najbardziej istotne cechy sygnału, skutecznie działając jako wieloskalowy, ułożony w stos autokoder odszumiający (stacked denoising autoencoder).2

W przeciwieństwie do standardowych głębokich sieci projektowanych z myślą o neuronauce z lat 60., PVM odzwierciedla bardziej współczesne odkrycia dotyczące w pełni rekurencyjnej natury kory nowej (neocortex).2 W hierarchii PVM jednostki niskiego poziomu przewidują zmiany na poziomie pikseli lub fragmentów obrazu (patch-level), podczas gdy jednostki wyższego poziomu wychwytują abstrakcyjne prawidłowości i dynamikę świata. Ponieważ sygnał uczący jest lokalny – pochodzi z różnicy między przewidywaniem jednostki a następującą po nim rzeczywistością – pozostaje on silny w całej hierarchii, niezależnie od głębokości sieci.2 Ta strukturalna decentralizacja pozwala systemowi być odpornym na zlokalizowane uszkodzenia, takie jak martwe piksele kamery, które są po prostu odrzucane w predykcjach na niższych poziomach i ignorowane przez reprezentacje wyższego rzędu.2

Matematyczna reprezentacja głównego mechanizmu PVM obejmuje minimalizację lokalnego błędu predykcji. Dla danego sygnału sensorycznego $x(t)$, jednostka predykcyjna $U_i$ generuje predykcję $\hat{x}_i(t+1)$ na podstawie swojego obecnego stanu wewnętrznego $s_i(t)$ i sprzężenia zwrotnego kontekstu $f_i(t)$. Błąd lokalny $E_i$ definiuje się jako:

$$E_i(t+1) = \|x_i(t+1) - \hat{x}_i(t+1)\|^2$$

Ten sygnał błędu jest używany do aktualizacji wag pamięci asocjacyjnej wewnątrz jednostki, zapewniając, że system stale dostosowuje się do obserwowanej dynamiki swojego środowiska.2 Zastosowanie lokalnych sygnałów uczących pozwala PVM ominąć złożoną propagację wsteczną w czasie (BPTT) wymaganą przez rekurencyjne sieci neuronowe (RNN) w tradycyjnych frameworkach, co czyni go naturalnie dopasowanym do asynchronicznej, opartej na zdarzeniach natury sprzętu neuromorficznego.4

| Cecha | Predictive Vision Model (PVM) | Standardowa konwolucyjna sieć neuronowa (CNN) |
| :--- | :--- | :--- |
| Reguła uczenia | Lokalny błąd predykcji (nienadzorowane) | Globalna propagacja wsteczna błędu (nadzorowane) |
| Skupienie czasowe | Z natury ciągłe i predykcyjne | Zazwyczaj statyczne lub sekwencyjne klatki |
| Łączność | W pełni rekurencyjna i wieloskalowa | Głównie jednokierunkowa (feedforward) z lokalnym poolingiem |
| Odporność | Odporność na lokalny szum i "martwe punkty" | Wrażliwość na przesunięcia rozkładu danych wejściowych |
| Skalowanie | Ograniczone przez lokalne obliczenia (Prawo Amdahla) | Ograniczone przez przepustowość pamięci i BPTT |

Architektura PVM pozwala na "swobodne łączenie" (liberal wiring) sprzężeń zwrotnych i połączeń bocznych.10 Jeśli sygnały te dostarczają wartości predykcyjnej, jednostki je integrują; w przeciwnym razie są one naturalnie ignorowane przez proces uczenia. Ta elastyczność umożliwia fuzję wielu modalności – takich jak wzrok, słuch i dotyk – na różnych poziomach abstrakcji, pozwalając im na wzajemne przewidywanie swoich przyszłych stanów.2

## Stan wiedzy w śledzeniu opartym na zdarzeniach

Krajobraz śledzenia obiektów przesuwa się obecnie z tradycyjnych kamer opartych na klatkach w stronę czujników neuromorficznych, w szczególności Dynamic Vision Sensors (DVS). Konwencjonalne kamery cierpią z powodu rozmycia ruchu (motion blur) przy dużych prędkościach i niskiego zakresu dynamiki w trudnych warunkach oświetleniowych, co często skutkuje niepowodzeniem śledzenia.9 Kamery zdarzeniowe (event cameras), które asynchronicznie raportują tylko zmiany intensywności dla poszczególnych pikseli, oferują wysoką rozdzielczość czasową (w skali mikrosekund) i wysoki zakres dynamiki (ponad 120 dB), co czyni je idealnymi do szybkiej percepcji robotycznej.8

Ostatnie osiągnięcia w tej dziedzinie wprowadziły paradygmat "Tracking Any Point" (TAP). Metody takie jak ETAP (Event-based Tracking of Any Point), zaprezentowane na CVPR 2025, wykazały, że wykorzystanie globalnego kontekstu obrazu pozwala na śledzenie półgęstych trajektorii w warunkach, w których metody oparte na klatkach zawodzą.13 ETAP wykorzystuje bloki uwagi przestrzennej i czasowej do iteracyjnej aktualizacji pozycji i wyglądu punktów, osiągając 135% poprawę w metryce Jaccarda w stosunku do istniejących rozwiązań bazowych.13 Jednak wiele z tych modeli SOTA (State of the Art) jest nadal projektowanych dla wysokiej klasy układów GPU, co zużywa znaczne ilości energii i ogranicza ich użyteczność w autonomicznych mikro-pojazdach powietrznych (MAV) lub urządzeniach brzegowych (edge devices).14

Inne znaczące podejście SOTA obejmuje wykorzystanie platform neuromorficznych, takich jak SpiNNaker, do śledzenia w czasie rzeczywistym. Systemy te wykorzystują rekurencyjne impulsowe sieci neuronowe (SNN) i mechanizmy dynamicznej uwagi do śledzenia obiektów, nawet gdy ich ruch zwalnia lub zatrzymuje się.17 Choć imponujące, systemom tym często brakuje ścisłej integracji i programowalnych modeli neuronów, które można znaleźć w architekturze Intel Loihi 2.6

| Model / Podejście | Platforma | Kluczowa zaleta | Ograniczenie |
| :--- | :--- | :--- | :--- |
| ETAP (CVPR 2025) | NVIDIA GPU | Globalny kontekst, śledzenie półgęste | Wysokie zużycie energii (~25W+) |
| SpiNNaker Tracker | SpiNNaker | Dynamiczna uwaga sterowana zdarzeniami | Specjalistyczny, niekomercyjny sprzęt |
| YOLO-SDNN | Loihi 2 / GPU | 12-krotna rzadkość operacji synaptycznych (SynOps) | Złożona faza treningu/konwersji |
| MCFNet | Hybrydowa (DVS/Klatki) | Wypaczanie (warping) oparte na przepływie optycznym | Wysoka intensywność obliczeniowa |
| PVM (Oryginał) | Klaster CPU | Wieloskalowe uczenie nienadzorowane | Powolna zbieżność na tradycyjnym sprzęcie |

Głównym wyzwaniem w tej sferze jest problem "zależności od ruchu": ponieważ czujniki DVS generują dane tylko wtedy, gdy występuje ruch, sygnał jest nierozerwalnie związany z trajektorią kamery.13 Wymaga to algorytmów, które są niezmiennicze (invariant) względem specyficznej dynamiki ruchu czujnika, przy jednoczesnym zachowaniu wysokiej wierności w reprezentacji obiektu. Zdolność PVM do traktowania zakłóceń czasowych jako "części rzeczywistości do przewidzenia" czyni go wyjątkowo odpornym na rzadką i nieregularną naturę danych zdarzeniowych.2

## Definicja problemu i luki sprzętowo-programowe

Pomimo teoretycznych zalet PVM, jego początkowe implementacje były ograniczone przez dostępną infrastrukturę obliczeniową z połowy lat 2010. Oryginalny kod był w dużej mierze oparty na języku Python, wykorzystując wielordzeniowe symulacje CPU, które wymagały tygodni do osiągnięcia zbieżności na danych o stosunkowo niskiej rozdzielczości.2 Chociaż później opracowano wersję akcelerowaną przez CUDA (PVM_cuda), zachowała ona wiele narzutów związanych z orkiestracją w Pythonie i nie była zoptymalizowana pod kątem specyficznych ograniczeń architektonicznych krzemu neuromorficznego.7

Ponadto problem "zanikającego gradientu" w głębokim uczeniu jest często rozwiązywany za pomocą "magicznych sztuczek", takich jak dropout, ReLU i specjalistyczny pooling.4 PVM unika ich z założenia dzięki lokalnym sygnałom błędu, ale ten rozproszony mechanizm treningowy nie mapuje się wydajnie na standardowe operacje oparte na tensorach we współczesnych frameworkach głębokiego uczenia (np. PyTorch, TensorFlow).4 Frameworki te są zoptymalizowane pod kątem synchronicznego, gęstego przetwarzania wsadowego (batch processing), co jest przeciwieństwem asynchronicznych, lokalnych i rzadkich aktualizacji wymaganych przez PVM.2

Istnieje również znacząca "ściana energetyczna" na brzegu sieci (edge). Obecne akceleratory AI dla urządzeń brzegowych, takie jak seria NVIDIA Jetson, oferują wysokie TOPS (Tera Operations Per Second), ale działają przy poziomach mocy (10-25W), które są nie do utrzymania dla małych platform autonomicznych.14 Sprzęt neuromorficzny, taki jak Loihi 2, który działa przy poziomach mocy poniżej 1W, stanowi rozwiązanie, ale wymaga fundamentalnej zmiany w projektowaniu algorytmów – przejścia z aktywacji zmiennoprzecinkowych na impulsy (spikes) oparte na zdarzeniach.5

## Proponowana metodologia: Strategia dwufazowej integracji

Metodologia projektu jest podzielona na dwie odrębne fazy techniczne: (1) Akceleracja wysokowydajnego treningu przy użyciu CUDA C oraz (2) Wdrożenie neuromorficzne za pośrednictwem frameworka Lava i procesora Intel Loihi 2. Podejście to uznaje, że chociaż sprzęt neuromorficzny jest lepszy do wnioskowania i adaptacji online, początkowy trening złożonych modeli hierarchicznych jest obecnie najbardziej wydajny na wysokiej klasy układach GPU.

### Faza 1: Wysokowydajne szkolenie w CUDA C

Aby przezwyciężyć wąskie gardło treningowe, projekt przepisze główną logikę PVM w języku CUDA C. Obejmuje to wyjście poza implementację PVM_cuda opartą na Pythonie do niskopoziomowej architektury jąder (kernels), która maksymalizuje przepustowość pamięci i wykorzystanie obliczeń.

*   **Równoległość jąder dla jednostek asocjacyjnych:** Każda jednostka PVM (koder predykcyjny) zostanie zmapowana do określonego bloku CUDA. Ponieważ jednostki PVM przetwarzają lokalne fragmenty pola widzenia, tę lokalność przestrzenną można wykorzystać do przechowywania danych w pamięci współdzielonej (shared memory), minimalizując kosztowne transakcje w pamięci globalnej.2
*   **Asynchroniczne aktualizacje wielopoziomowe:** Ułożona w stos architektura PVM pozwala warstwom działać w różnych skalach czasowych. CUDA C pozwala na użycie wielu strumieni (streams) do nakładania na siebie obliczeń różnych warstw, zapewniając, że GPU pozostaje nasycone nawet podczas obliczania reprezentacji wyższego rzędu.7
*   **Zoptymalizowane obliczanie lokalnej straty:** Lokalny błąd $E_i$ będzie obliczany w miejscu (in-place) podczas przejścia w przód (forward pass), a aktualizacje wag będą stosowane natychmiast w ramach tego samego wykonania jądra, odzwierciedlając naturę online PVM i zmniejszając potrzebę wielokrotnego przechodzenia przez dane.2

Oczekuje się, że to wysokowydajne przepisanie kodu skróci czas zbieżności dla modeli na dużą skalę z miesięcy do godzin, umożliwiając szybką iterację wymaganą do osiągnięcia wydajności SOTA.2

### Faza 2: Mapowanie do frameworka Lava i Loihi 2

Po fazie treningu parametry modelu zostaną przeniesione do frameworka Lava. Lava to framework open-source przeznaczony do tworzenia aplikacji inspirowanych układem nerwowym, zapewniający modułową strukturę do mapowania procesów na różne backendy, w tym CPU, GPU i chipy Loihi.18

*   **Transformacja Sigma-Delta:** Aktywacje PVM zostaną przekształcone w sieci neuronowe Sigma-Delta (SDNN). W SDNN neurony odpalają impulsy tylko wtedy, gdy zmiana ich aktywacji przekracza zdefiniowany próg.5 Odpowiada to zasadzie PVM polegającej na przewidywaniu zmian i przesyłaniu tylko reszt (błędów).2
*   **Kwantyzacja stałoprzecinkowa i impulsy stopniowane (Graded Spikes):** Loihi 2 obsługuje impulsy stopniowane z maksymalnie 32-bitowymi ładunkami całkowitymi (integer payloads).6 Projekt wykorzysta te ładunki do reprezentowania skwantyzowanej precyzji reszt predykcyjnych PVM, zapewniając, że dokładność nie zostanie poświęcona na rzecz rzadkości (sparsity).5
*   **Uczenie na chipie (On-Chip Learning) za pośrednictwem Lava:** Projekt zaimplementuje lokalne reguły uczenia PVM przy użyciu `Loihi3FLearningRule` w ramach Lava.21 Pozwala to PVM na ciągłe dostosowywanie się do nowych środowisk wizualnych nawet po wdrożeniu na procesorze Loihi 2.6

Konwersja z modelu wytrenowanego w CUDA C do Lava wykorzysta bibliotekę `lava-dl.netx`, która automatyzuje generowanie procesów Lava z opisów sieci.22

## Unikalna wartość architektury Intel Loihi 2

Intel Loihi 2 stanowi kamień milowy w krzemie neuromorficznym, oferując 128 w pełni asynchronicznych rdzeni neuronowych i obsługując do 1 miliona neuronów oraz 120 milionów synaps na chip.6 Jego architektura jest wyjątkowo dostosowana do PVM z kilku powodów:

*   **Przetwarzanie blisko pamięci (Near-Memory Computing):** Loihi 2 umieszcza obliczenia i pamięć w tych samych rdzeniach neuronowych, omijając wąskie gardło von Neumanna. Dla PVM, który wymaga częstego dostępu do wag i stanów tysięcy lokalnych jednostek asocjacyjnych, architektura ta znacznie zmniejsza zużycie energii i opóźnienia.23
*   **Programowalne modele neuronów:** W przeciwieństwie do swojego poprzednika, Loihi 2 posiada programowalny potok w każdym rdzeniu neuromorficznym.6 Pozwala to na implementację specyficznej nieliniowej dynamiki wymaganej przez jednostki pamięci asocjacyjnej PVM, w tym niestandardowych progów i funkcji resetowania.18
*   **Asynchroniczne projektowanie obwodów:** Przeprojektowane asynchroniczne obwody chipu zapewniają do 10-krotnego przyspieszenia przetwarzania impulsów w stosunku do pierwszej generacji.6 Prędkość ta jest krytyczna dla śledzenia w czasie rzeczywistym szybko poruszających się obiektów wykrywanych przez kamery DVS.25
*   **Technologia procesu Intel 4:** Wyprodukowany w procesie Intel 4, Loihi 2 osiąga znaczny wzrost gęstości i szybkości tranzystorów, z krokami czasowymi dla całego chipu wynoszącymi zaledwie 200 ns.6

| Specyfikacja | Loihi 1 | Loihi 2 | Korzyść dla PVM |
| :--- | :--- | :--- | :--- |
| Proces | 14nm | Intel 4 | Gęstsza integracja hierarchii PVM |
| Maks. liczba neuronów | 128,000 | 1,000,000 | Obsługa wizji o wysokiej rozdzielczości |
| Ładunek impulsu (Spike Payload) | Binarny (1-bit) | Stopniowany (do 32-bit) | Reszty predykcyjne o wysokiej precyzji |
| Model neuronu | Stały LIF | W pełni programowalny | Konfigurowalne pamięci asocjacyjne |
| Reguły uczenia | 2-czynnikowe + nagroda | 3-czynnikowe + lokalna modyfikacja | Adaptacja na chipie w czasie rzeczywistym |

Możliwość "impulsów stopniowanych" (graded spike) jest być może najbardziej krytyczna dla PVM. Ponieważ PVM opiera się na błędach predykcji, które mogą przyjmować dowolną wartość, binarne impulsy wymagałyby nieefektywnego kodowania częstotliwościowego (rate-coding), prowadząc do wysokich opóźnień. Impulsy stopniowane pozwalają na bezpośrednią transmisję wielkości błędu w pojedynczym zdarzeniu, zachowując matematyczną integralność PVM przy jednoczesnym utrzymaniu neuromorficznej rzadkości.5

## Porównanie z wcześniejszymi pracami i kontekst uczestników

Proponowane podejście jest bezpośrednią ewolucją badań prowadzonych przez Filipa Piekniewskiego i współpracowników w BrainCorp, którzy po raz pierwszy wprowadzili PVM w 2016 roku.3 Oryginalna praca wykazała, że PVM może osiągnąć imponującą wydajność śledzenia na filmach o niskiej rozdzielczości (96x96) bez potrzeby etykietowanych danych treningowych lub specjalistycznych technik regularyzacji (np. dropout, batch norm) stosowanych w tradycyjnym głębokim uczeniu.2

Jednak pierwotni uczestnicy zauważyli, że rekurencyjne sprzężenie zwrotne i działanie online PVM sprawiły, że jego skuteczna implementacja w standardowych frameworkach głębokiego uczenia była trudna, jeśli nie niemożliwa.10 Późniejsze przesunięcie się dziedziny w kierunku układów GPU i wielkoskalowego uczenia nadzorowanego pozostawiło potencjał PVM dla ultra-energooszczędnych obliczeń brzegowych w dużej mierze niezbadanym. Ten projekt powraca do tych fundamentalnych idei, wykorzystując obecnie dojrzały framework Lava i sprzęt Loihi 2, aby zrealizować pierwotną wizję skalowalnego, inspirowanego biologicznie systemu wizyjnego.3

## Powiązanie z publikacjami INRC

Społeczność Intel Neuromorphic Research Community (INRC) opublikowała kilka ilościowych benchmarków, które uzasadniają przejście na Loihi 2 w zadaniach wizyjnych. Na przykład "CarSNN" i "LaneSNNs" zademonstrowały wydajną autonomiczną jazdę opartą na zdarzeniach na Loihi, podczas gdy inne badania wykazały, że Loihi replikuje klasyczne symulacje neuronowe z wysoką precyzją i doskonałą skalowalnością.27

| Badanie | Główny wniosek | Znaczenie dla PVM |
| :--- | :--- | :--- |
| LCA na Loihi 2 18 | Konkurencyjne wyniki w zadaniach rzadkiego kodowania (sparse coding) | Waliduje kodowanie predykcyjne na Loihi |
| SDNN PilotNet 22 | 12-krotna rzadkość SynOps w stosunku do ANN | Dowód słuszności koncepcji (PoC) dla wizji Sigma-Delta |
| VOD z Resonator 27 | Wydajna odometria wizualna | Potwierdza opłacalność złożonych modeli czasowych |
| Loihi vs. Jetson 29 | 250x mniej energii niż Orin Nano | Ustanawia cel efektywności energetycznej |

Obecny zespół ma wyjątkową pozycję, aby odnieść sukces, ponieważ posiada głęboką wiedzę architektoniczną na temat lokalnych reguł uczenia PVM oraz wiedzę specjalistyczną pozwalającą na poruszanie się po specyficznych wymaganiach mikrokodu rdzeni Loihi 2.2

## Ocena ilościowa i metryki wydajności

Sukces projektu zostanie zmierzony poprzez analizę porównawczą zarówno z oryginalnym PVM, jak i współczesnymi trackerami SOTA (np. ETAP, YOLO-SDNN) na standardowych zestawach danych opartych na zdarzeniach.13

### Zestawy danych do ewaluacji

*   **N-ImageNet:** Używany do testowania drobnoziarnistego rozpoznawania i uczenia reprezentacji ze strumieni zdarzeń.32
*   **Gen1 Detection Dataset:** Zapewnia 39 godzin nagrań motoryzacyjnych do oceny dokładności lokalizacji i śledzenia obiektów w rzeczywistym ruchu drogowym.31
*   **DSEC-Det:** Zestaw danych o szybkim ruchu drogowym używany do testowania wydajności w trudnych warunkach oświetleniowych i scenariuszach z szybko poruszającymi się obiektami.16

### Metryki ilościowe

Projekt zastosuje wielowymiarowe ramy ewaluacji:

*   **Dokładność śledzenia:** Wskaźnik sukcesu i precyzja zdefiniowane przez metrykę Jaccarda i mAP (mean Average Precision). Sukces definiuje się jako osiągnięcie wyniku w granicach 5% od nadzorowanych modeli SOTA (np. ETAP) przy jednoczesnym zachowaniu nienadzorowanego reżimu treningowego.13
*   **Opóźnienie (Latency):** Czas od wystąpienia zdarzenia zmiany jasności do aktualizacji trajektorii obiektu. Celem jest czas reakcji w granicach 2,5 ms, dorównujący możliwościom neuromorficznych systemów szybkiego rozpoznawania SOTA.25
*   **Efektywność energetyczna:** Mierzona w kategoriach iloczynu energii i opóźnienia (Energy-Delay Product - EDP) oraz zużycia mocy. Celem jest >50-krotna poprawa wydajności w porównaniu do NVIDIA Jetson Orin Nano z uruchomionym standardowym trackerem.14
*   **Rzadkość obliczeniowa (Computational Sparsity):** Kwantyfikowana przez liczbę operacji synaptycznych (SynOps) na wnioskowanie w porównaniu do operacji mnożenia i akumulacji (MAC) w sztucznej sieci neuronowej (ANN) o izo-architekturze.22

| Platforma | Model | Moc | Opóźnienie | Cel efektywności |
| :--- | :--- | :--- | :--- | :--- |
| NVIDIA Jetson Orin Nano | YOLOv8 / ETAP | 10-25W | 25-40 ms | Wartość bazowa (Baseline) |
| Intel Loihi 2 | SDNN-PVM (Cel) | <1W | <5 ms | >100x Poprawa |
| SpiNNaker | Rekurencyjna SNN | ~10W | <10 ms | Baza do porównania |

Sukces definiuje się jako udaną implementację wielopoziomowego PVM, który potrafi śledzić obiekty w zestawie danych Gen1 z mAP wynoszącym co najmniej 0,40, zużywając przy tym mniej niż 500 mW mocy.25

## Definiowanie sukcesu i długoterminowy wpływ

Pomyślny wynik tego projektu wykaże, że Predictive Vision Model, zaimplementowany na sprzęcie neuromorficznym, stanowi realną alternatywę dla energochłonnych modeli głębokiego uczenia, które obecnie dominują w tej dziedzinie. Oznaczałoby to pierwszy przypadek, w którym w pełni rekurencyjna, nienadzorowana architektura kodowania predykcyjnego została pomyślnie wdrożona na dużą skalę w krzemie.2

### Wpływ techniczny

*   **Ominięcie ściany energetycznej:** Udowodnienie, że złożone hierarchie wizualne mogą działać w zakresie miliwatów, otwiera drzwi dla wszechobecnych systemów autonomicznych, od robotów magazynowych po drony do monitorowania ekologicznego.23
*   **Rozwój uczenia nienadzorowanego:** Walidacja zdolności PVM do budowania użytecznych reprezentacji bez ludzkiego etykietowania rozwiązuje jedno z głównych wąskich gardeł w skalowaniu AI do nowych domen.1
*   **Wkład w ekosystem Lava:** Zoptymalizowane procesy PVM i implementacje lokalnych reguł uczenia zostaną wniesione z powrotem do frameworka Lava, zapewniając szablon dla innych badaczy INRC do implementacji hierarchicznych modeli predykcyjnych.20

### Wpływ społeczny i przemysłowy

Szerszy wpływ obejmuje przejście do "Sztucznej Inteligencji Następnej Generacji", która jest mniej "krucha" i bardziej zdolna do radzenia sobie z niepewnością i wieloznacznością świata naturalnego.1 Poprzez emulację struktury neuronowej i działania ludzkiego mózgu, neuromorficzne systemy PVM będą lepiej przygotowane do radzenia sobie z nowymi sytuacjami, takimi jak nieoczekiwane zmiany oświetlenia lub okluzje, bez katastrofalnych awarii.1 Jest to krytyczne dla aplikacji o znaczeniu krytycznym dla bezpieczeństwa, takich jak autonomiczna jazda i robotyka poszukiwawczo-ratownicza.1

## Logistyka wdrożenia: Potok od CUDA C do Lava

Projekt będzie przestrzegał rygorystycznego potoku inżynierii oprogramowania, aby zapewnić niezawodność przejścia między backendami.

*   **Rozwój w CUDA C:** Jądra PVM będą rozwijane przy użyciu środowiska kontenerowego (Docker/NVIDIA-Docker), aby zapewnić powtarzalność i łatwy dostęp do sterowników GPU.7
*   **Weryfikacja z PVM w Pythonie:** Początkowe wyniki w CUDA C zostaną zweryfikowane z istniejącym kodem PVM_cuda, aby upewnić się, że matematyczna logika jednostek asocjacyjnych i wieloskalowa hierarchia pozostają nienaruszone.7
*   **Symulacja w Lava:** Przed wdrożeniem sprzętowym, PVM-SDNN będzie symulowany przy użyciu backendów CPU/GPU frameworka Lava. Pozwala to na precyzyjne dostrojenie progów i precyzji bitowej bez ograniczeń związanych z dostępem do fizycznego sprzętu.18
*   **Wykonanie sprzętowe:** Ostateczne wdrożenie na Loihi 2 wykorzysta płyty Intel Kapoho Point (8-chipowe) lub VPX, w zależności od wymaganej skali modelu.5

Implementacja aktualizacji potencjału błonowego dla neuronów Loihi 2, wzorowana na równaniach Leaky Integrate and Fire (LIF), będzie podążać za standardową całkowaniem numerycznym Eulera pierwszego rzędu z precyzją całkowitoliczbową 28:

$$v[t] = v[t-1] \cdot (2^{12} - \delta_u) \cdot 2^{-12} + I[t] + I_{bias}$$

gdzie $v[t]$ to potencjał błonowy, $\delta_u$ to stała zaniku (decay constant), a $I[t]$ to wejście synaptyczne.30 To formalne przestrzeganie wewnętrznej dynamiki Loihi zapewnia, że implementacja PVM jest w pełni natywna dla sprzętu, maksymalizując zarówno szybkość, jak i wydajność.

## Wnioski: Ku nowemu paradygmatowi w wizji robotycznej

Integracja modelu Predictive Vision Model z frameworkiem Lava i sprzętem Loihi 2 rozwiązuje fundamentalne ograniczenia współczesnego śledzenia obiektów: kompromis między dokładnością, opóźnieniem i mocą. Wykorzystując nienadzorowaną architekturę, która uczy się nieodłącznych prawidłowości świata wizualnego, projekt ten odchodzi od "kruchych" modeli deterministycznych z przeszłości w kierunku bardziej odpornej i elastycznej formy AI.1

Unikalna synergia między lokalnymi sygnałami błędu PVM a asynchroniczną architekturą Loihi 2, przetwarzającą blisko pamięci, zapewnia ilościową ścieżkę naprzód dla obliczeń brzegowych. Sukces w tym projekcie nie tylko posunie naprzód stan wiedzy w śledzeniu opartym na zdarzeniach, ale także zapewni fundamentalną metodologię wdrażania w pełni rekurencyjnych, inspirowanych biologicznie systemów w rzeczywistych aplikacjach autonomicznych.1 Połączenie CUDA C do szybkiego treningu i Lava do wydajnego wdrażania stanowi solidną i skalowalną strategię dla przyszłości percepcji neuromorficznej.

## Cytowane prace

1. Neuromorphic Computing - Next Generation of AI - Intel, otwierano: lutego 26, 2026, https://www.intel.la/content/www/xl/es/research/neuromorphic-computing.html
2. Predictive Vision in a nutshell – Piekniewski's blog, otwierano: lutego 26, 2026, https://blog.piekniewski.info/2016/11/04/predictive-vision-in-a-nutshell/
3. July 2016 - Piekniewski's blog, otwierano: lutego 26, 2026, https://blog.piekniewski.info/2016/07/
4. Predictive Learning [pdf] | Hacker News, otwierano: lutego 26, 2026, https://news.ycombinator.com/item?id=13120794
5. Sigma-Delta Neural Network Conversion on Loihi 2 - arXiv, otwierano: lutego 26, 2026, https://arxiv.org/html/2505.06417v1
6. Taking Neuromorphic Computing with Loihi 2 to the Next Level Technology Brief - Intel, otwierano: lutego 26, 2026, https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf
7. piekniewski/PVM_cuda: Cuda implementation of Predictive ... - GitHub, otwierano: lutego 26, 2026, https://github.com/piekniewski/PVM_cuda
8. Robust Event-Based Object Tracking Combining Correlation Filter and CNN Representation - PMC, otwierano: lutego 26, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC6795673/
9. Hardware, Algorithms, and Applications of the Neuromorphic Vision Sensor: A Review, otwierano: lutego 26, 2026, https://www.mdpi.com/1424-8220/25/19/6208
10. PVM is out - Piekniewski's blog, otwierano: lutego 26, 2026, https://blog.piekniewski.info/2016/07/26/pvm-is-out/
11. Loihi: A Neuromorphic Manycore Processor with On-Chip Learning, otwierano: lutego 26, 2026, https://redwood.berkeley.edu/wp-content/uploads/2021/08/Davies2018.pdf
12. Event-based solutions for human-centered applications: a comprehensive review - Frontiers, otwierano: lutego 26, 2026, https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1585242/full
13. CVPR Poster ETAP: Event-based Tracking of Any Point, otwierano: lutego 26, 2026, https://cvpr.thecvf.com/virtual/2025/poster/33145
14. Comparing NVIDIA Orin for Edge AI Performance | Things Embedded USA, otwierano: lutego 26, 2026, https://things-embedded.com/us/white-paper/deploying-ai-at-the-edge-with-nvidia-orin/
15. Benchmarking YOLOv8 Variants for Object Detection Efficiency on Jetson Orin NX for Edge Computing Applications - MDPI, otwierano: lutego 26, 2026, https://www.mdpi.com/2073-431X/15/2/74
16. N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras, otwierano: lutego 26, 2026, https://www.researchgate.net/publication/359005267_N-ImageNet_Towards_Robust_Fine-Grained_Object_Recognition_with_Event_Cameras
17. CVPR Event-Driven Dynamic Attention for Multi-Object Tracking on Neuromorphic Hardware, otwierano: lutego 26, 2026, https://cvpr.thecvf.com/virtual/2025/35593
18. A Look at Loihi 2 - Intel - Open Neuromorphic, otwierano: lutego 26, 2026, https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/
19. Performance Comparison Of Mobile GPUS For Object Detection In Edge Computing - Digital Commons @PVAMU - Prairie View A&M University, otwierano: lutego 26, 2026, https://digitalcommons.pvamu.edu/cgi/viewcontent.cgi?article=2655&context=pvamu-theses
20. Lava Software Framework — Lava documentation, otwierano: lutego 26, 2026, https://lava-nc.org/
21. lava/tutorials/in_depth/clp/tutorial01_one-shot_learning_with_novelty_detection.ipynb at main - GitHub, otwierano: lutego 26, 2026, https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/clp/tutorial01_one-shot_learning_with_novelty_detection.ipynb
22. lava-nc/lava-dl: Deep Learning library for Lava - GitHub, otwierano: lutego 26, 2026, https://github.com/lava-nc/lava-dl
23. Exploring Neuromorphic Computing with Loihi-2 for High-Performance CFD Simulations, otwierano: lutego 26, 2026, https://www.researchgate.net/publication/395305489_Exploring_Neuromorphic_Computing_with_Loihi-2_for_High-Performance_CFD_Simulations
24. Neuromorphic Computing 2025: Current SotA - human / unsupervised, otwierano: lutego 26, 2026, https://humanunsupervised.com/papers/neuromorphic_landscape.html
25. High-Speed Object Recognition Based on a Neuromorphic System - MDPI, otwierano: lutego 26, 2026, https://www.mdpi.com/2079-9292/11/24/4179
26. Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook - Dynamic field theory, otwierano: lutego 26, 2026, https://dynamicfieldtheory.org/upload/file/1631291311_c647b66b9e48f0a9baff/DavisEtAl2021.pdf
27. About the INRC - Confluence, otwierano: lutego 26, 2026, https://intel-ncl.atlassian.net/wiki/spaces/INRC/pages/76382230/INRC+Publications
28. Mapping and Validating a Point Neuron Model on Intel's Neuromorphic Hardware Loihi, otwierano: lutego 26, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC9197133/
29. A Complete Pipeline for deploying SNNs with Synaptic Delays on Loihi 2 - arXiv.org, otwierano: lutego 26, 2026, https://arxiv.org/html/2510.13757v1
30. Brian2Loihi: An emulator for the neuromorphic chip Loihi using the spiking neural network simulator Brian - Frontiers, otwierano: lutego 26, 2026, https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.1015624/full
31. TUMTraf EMOT: Event-Based Multi-Object Tracking Dataset and Baseline for Traffic Scenarios - arXiv, otwierano: lutego 26, 2026, https://arxiv.org/html/2512.14595v2
32. N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras, otwierano: lutego 26, 2026, https://82magnolia.github.io/n_imagenet/
33. [PDF] N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras, otwierano: lutego 26, 2026, https://www.semanticscholar.org/paper/231a013defe6ad8b6d6039ed477989670a3dc3f5
34. End-to-End Edge Neuromorphic Object Detection System | Request PDF - ResearchGate, otwierano: lutego 26, 2026, https://www.researchgate.net/publication/382405283_End-to-End_Edge_Neuromorphic_Object_Detection_System
35. Who will figure out intelligence? - Piekniewski's blog, otwierano: lutego 26, 2026, https://blog.piekniewski.info/2017/03/27/who-will-figure-out-intelligence/
36. About the INRC - Confluence, otwierano: lutego 26, 2026, https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview