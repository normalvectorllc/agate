FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up directory structure
WORKDIR /workspaces

# # Configure git for a better experience
# RUN git config --system core.editor "code --wait" \
#     && git config --system pull.rebase false

# Ensure everything is owned by vscode
USER vscode