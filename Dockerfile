FROM python:3.10
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 8501
CMD streamlit run src/app.py