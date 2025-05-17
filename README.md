Guess these steps are enough to setup your instance on http://127.0.0.1:5000

You probably can try it out on https://cv.ioann44.ru :)

```bash
python -m venv .venv # optional
source .venv/bin/activate # optional

pip install resources/requirements.txt

python server.py >> logs.txt 2>&1
```