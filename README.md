# Projet2_Disneyland

# Construire et lancer image depuis dockerfile
docker build -t fastdisneyland .
docker run -d --name fastdisneylandcontenair -p 8000:8000 fastdisneyland

# entrer dans container
docker exec -it myimage /bin/bash

# rm all images not start 
docker rm -vf $(docker ps -aq)
