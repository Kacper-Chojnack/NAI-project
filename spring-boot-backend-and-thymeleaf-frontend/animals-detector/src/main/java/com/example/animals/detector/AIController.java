package com.example.animals.detector;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Controller
@RequestMapping("/")
public class AIController {

    private final String pythonServerUrl = "http://localhost:5000/predict"; // Adres serwera Python

    @GetMapping
    public String home() {
        return "index"; // Wyświetlamy stronę główną (index.html)
    }

    @PostMapping("/api/predict")
    public String predict(@RequestParam("file") MultipartFile file, Model model) {
        if (file.isEmpty()) {
            model.addAttribute("error", "Nie wysłano pliku"); // Błąd: brak pliku
            return "index"; // Powrót do strony głównej
        }

        try {
            // Przygotowanie nagłówków do żądania
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // Zamiana pliku na ByteArrayResource
            ByteArrayResource fileResource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename(); // Nazwa oryginalnego pliku
                }
            };

            // Tworzymy ciało żądania z plikiem
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", fileResource);

            // Tworzymy żądanie HTTP
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // Wysyłamy żądanie do serwera Python
            RestTemplate restTemplate = new RestTemplate();
            ResponseEntity<Map> response = restTemplate.postForEntity(pythonServerUrl, requestEntity, Map.class);

            // Wyniki predykcji przekazujemy do widoku
            model.addAttribute("results", response.getBody());
            return "index"; // Powrót do strony z wynikami
        } catch (Exception e) {
            // Obsługa błędów
            model.addAttribute("error", "Błąd przetwarzania pliku: " + e.getMessage());
            return "index"; // Powrót do strony głównej
        }
    }

}
