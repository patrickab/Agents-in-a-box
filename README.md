``` bash
# Setup

sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  gnupg \
  lsb-release

curl -fsSL https://get.docker.com | sudo sh

sudo systemctl enable docker
sudo systemctl start docker

sudo usermod -aG docker $USER
newgrp docker

# Test
docker ps
```
