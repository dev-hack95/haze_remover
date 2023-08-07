FROM python:3.10
WORKDIR /app
COPY . /app
RUN apt update -y
RUN apt install apt-utils libgl1-mesa-glx -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 8501
CMD streamlit run src/app.py