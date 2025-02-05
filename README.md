# Klasyfikator Obrazów AI

## Cel projektu
Celem projektu jest stworzenie aplikacji, która wykorzystuje modele sztucznej inteligencji do rozpoznawania zwierząt na zdjęciach. Aplikacja działa jako połączenie frontendowej aplikacji webowej, serwera oraz modeli AI. Kluczowe elementy projektu:
- Zastosowanie istniejących modeli AI (bez trenowania od zera).
- Porównanie ich skuteczności na zbiorze testowym.
- Wykorzystanie AI jako praktycznego narzędzia w aplikacji.

## Funkcjonalności
- Wczytanie zdjęcia przez użytkownika w aplikacji webowej.
- Rozpoznanie zwierzęcia za pomocą trzech modeli: MobileNetV2, NASNetMobile, Xception.
- Wyświetlenie wyniku dla każdego modelu wraz z prawdopodobieństwem.

## Wykorzystane technologie
- **Frontend:** Thymeleaf + CSS.
- **Backend:** Spring Boot (Java) do obsługi żądań i komunikacji z serwerem AI.
- **Serwer AI:** Flask (Python) z zaimplementowanymi modelami AI.
- **Modele AI:** MobileNetV2, NASNetMobile, Xception.

## Zbiór danych
Do testowania i weryfikacji wykorzystano zbiór danych ze zdjęciami zwierząt dostępny na Kaggle: [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals).

- **Zbiór treningowy:** Zdjęcia zwierząt z kategorii pies, kot i inne.
- **Zbiór testowy:** Osobny zestaw zdjęć zawierający różnorodne zwierzęta (po 3 na kategorię).

## Jak uruchomić projekt
### 1. Klonowanie repozytorium
git clone https://github.com/Kacper-Chojnack/NAI-project.git

### 2. Instalacja zależności
cd python-backend/src
pip install -r requirements.txt

### 3. Odpalenie serwerów
cd python-backend/src
python server.py

cd spring-boot-backend-and-thymeleaf-frontend/animals-detector
mvn spring-boot:run

### 3. Dostęp do aplikacji
Aplikacja będzie dostępna pod adresem: http://localhost:8080.

