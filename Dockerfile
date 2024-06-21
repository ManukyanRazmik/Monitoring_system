FROM python:3.11

WORKDIR /server

# COPY requirements.txt .

# RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

EXPOSE 8000

CMD ["python", "impact2_engine/API.py"]