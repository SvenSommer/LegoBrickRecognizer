# Docker deploy instructions

1. Install docker ubuntu package
You can install docker by the following shell commands:

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y curl
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce

2. Build docker image
sudo docker build -t lego_brick_recognition_server .

3. Run server solution:
sudo docker run -p 5001:5001 --name LegoBrickServer -d lego_brick_recognition_server

(Optional) If you need see server logs use the following command:
sudo docker run -p 5001:5001 --name LegoBrickServer -it lego_brick_recognition_server

4. To delete the containse use the following command:
sudo docker rm LegoBrickServer

5. (Optional) You can test server by the following shell command:
time curl -H "Content-Type: application/json" -X POST -d '{"url":"https://i.ebayimg.com/00/s/NzU4WDExNDg=/z/OKoAAOSwwNdgcFmR/$_59.JPG"}' http://localhost:5001/api/inference/url
