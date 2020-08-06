docker run --gpus 1 --shm-size=40G \
    -v `pwd`:/home/${USER}/project \
    -p 8888:8888 \
    --name ${USER}.kaggle \
    -itd kaggle

