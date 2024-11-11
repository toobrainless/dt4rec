#!/bin/bash

# Запуск train.py с разными аргументами
python3 train.py emb_64_traj_100_default False False
python3 train.py emb_64_traj_100_svd_freeze True False
python3 train.py emb_64_traj_100_svd_unfreeze True True
# Добавь столько команд, сколько нужно

# Выводим сообщение об окончании всех запусков
echo "Все задачи выполнены!"