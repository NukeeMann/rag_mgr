FROM python:3.10.15

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN set -x && \
    apt-get update && \
    apt-get install --no-install-recommends --assume-yes \
      build-essential \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create user
ARG UID=1000
ARG GID=1000
RUN set -x && \
    groupadd -g "${GID}" python && \
    useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python &&\
    chown python:python -R /app

USER python

# Install python dependencies
COPY requirements.txt .
RUN  pip install --no-cache-dir --upgrade -r requirements.txt
RUN  python -m spacy download pl_core_news_sm
# Copy application code
COPY src src

# don't buffer Python output
ENV PYTHONUNBUFFERED=1
# Add pip's user base to PATH
ENV PATH="$PATH:/home/python/.local/bin"

# expose port
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/main.py", "--server.port", "8501"]