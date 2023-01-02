FROM inseefrlab/onyxia-rstudio:ds-r4.2.3

# Install packages specified in the renv.lock file
RUN git clone https://github.com/ThomasFaria/DT-RN-chapitre1.git && \
    cd DT_RN_chapitre1 && \
    Rscript -e "renv::restore()" && \
    chown -R ${USERNAME}:${GROUPNAME} ${HOME}
    
SHELL ["/bin/bash", "-c"]

ARG PYTHON_VERSION="3.10.4"
ENV MAMBA_DIR="/opt/mamba"
ENV PATH="${MAMBA_DIR}/bin:${PATH}"

# Installation de magick
RUN git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.0 && \
    cd ImageMagick-7.1.0 && \
    ./configure && \
    make && \
    sudo make install && \
    sudo ldconfig /usr/local/lib

COPY conda-env.yml .

# Install minimal python
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O mambaforge.sh && \
    # Install mambaforge latest version
    /bin/bash mambaforge.sh -b -p "${MAMBA_DIR}" && \
    # Set specified Python version in base Conda env
    mamba install python=="${PYTHON_VERSION}" && \
    # Install essential Python packages
    mamba env update -n base -f conda-env.yml && \
    # Activate custom Conda env by default in shell
    echo ". ${MAMBA_DIR}/etc/profile.d/conda.sh && conda activate" >> ${HOME}/.bashrc && \
    # fix for version GLIBCXX_3.4.30
    sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
    sudo ln -s /opt/mamba/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
    # Fix permissions
    chown -R ${USERNAME}:${GROUPNAME} ${HOME} ${MAMBA_DIR} && \
    # Clean
    rm mambaforge.sh conda-env.yml && \ 
    mamba clean --all -f -y

CMD ["python3"]
