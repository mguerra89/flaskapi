FROM python:3.7.9

WORKDIR /usr/src/app

COPY requiretments.txt ./
COPY app.py ./
COPY class_map.pkl ./
COPY model.h5 ./

RUN pip install --no-cache-dir -r requiretments.txt
COPY . .

CMD [ "python", "./app.py" ]

EXPOSE 5000