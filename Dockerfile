FROM python:3

COPY . /app

WORKDIR  /app

RUN apt-get update
RUN apt-get install -y python3
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1
RUN pip install llama-cpp-python
EXPOSE 8000

CMD ["chainlit","run","app.py"]