FROM python:3.12-bookworm

# 1. System deps: mergerfs + fuse (disk pooling), espeak-ng (phonemizer),
#    ffmpeg (torchaudio backend), build-essential (pyworld C++ ext), libsndfile1 (soundfile)
RUN apt-get update && apt-get install -y \
    mergerfs \
    fuse \
    espeak-ng \
    ffmpeg \
    build-essential \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Poetry
RUN pip install --no-cache-dir poetry==2.2.1

# 3. Configure Poetry to not create virtual environments 
RUN poetry config virtualenvs.create false

# 4. Set the working directory
WORKDIR /workspace

# 5. Pre-install dependencies to cache the Docker layer
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# 6. Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
