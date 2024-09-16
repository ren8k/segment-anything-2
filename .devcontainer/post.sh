#!/bin/bash
pip install -e ".[demo]"
cd checkpoints && \
./download_ckpts.sh && \
cd ..
