# face-health-ai Prototype

Минимальный прототип для хакатона: FastAPI-бэкенд с React-интерфейсом (через CDN), который делает снимок с веб-камеры пользователя и отправляет его на анализ. Детектор основан на предобученном каскаде Хаара из OpenCV.

## Стек

- **Backend:** Python 3.11, FastAPI, OpenCV Haar Cascade
- **Frontend:** React 18 (UMD), vanilla CSS, WebRTC getUserMedia

## Запуск

1. Установите зависимости бэкенда и запустите сервер:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

2. Откройте `http://127.0.0.1:8000` в браузере. Разрешите доступ к камере, нажмите «Запустить камеру», затем «Снимок и анализ». Ответ содержит число найденных лиц и их координаты.

## Структура проекта

```
backend/
  app/
    main.py        # FastAPI приложение с эндпоинтом /api/analyze
  requirements.txt
frontend/
  app.js           # Минималистичный React-компонент
  styles.css       # Стили интерфейса
```

## Дальнейшее развитие

- Добавить дополнительные маркеры здоровья на основе найденных лиц (цвет кожи, симметрия и т.д.).
- Сохранение истории анализов и авторизация пользователей.
- Мобильный клиент на React Native или Flutter.
- Поддержка моделей глубокого обучения (например, RetinaFace) и анализ мимики.
