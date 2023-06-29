FROM python:3.10

WORKDIR /app

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python3 -c "import nltk; nltk.download('punkt'); exit()"

WORKDIR /app/preprocess

RUN python3 dialogue_segmentation.py

CMD [ "python3", "dialogue_segmentation.py" ]
