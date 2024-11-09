#!/bin/bash

# Запуск train.py с разными аргументами
python3 train.py old_attn_emb_64 False False True False
python3 train.py new_attn_emb_64 False False False False
python3 train.py old_attn_emb_128 False False True True
python3 train.py new_attn_emb_128 False False False True
# Добавь столько команд, сколько нужно

# Выводим сообщение об окончании всех запусков
echo "Все задачи выполнены!"