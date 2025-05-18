Если повезёт
```bash
docker compose up -d
```

Если нет, и нужно пересобирать всё заново (например, после изменения `requirements.txt`)
```bash
docker build . --build --no-cache
```

Пересобрать только со свежим `git clone`, зависимости не менялись
```bash
docker build --build-arg GIT_CLONE_INVALIDATE=$(date +%s) .
```