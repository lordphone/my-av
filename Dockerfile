FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml ./
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "ml", "/bin/bash", "-c"]

COPY . .

CMD ["conda", "run", "--no-capture-output", "-n", "ml", "bash", "scripts/run_training.sh"]
