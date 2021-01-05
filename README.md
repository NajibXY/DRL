Auteur
===========
Najib EL KHADIR - 1511175 - M2 IA 2020/2021 - Université Lyon 1

Dépendances
===========
* python3+
* pipenv
* packages nécessaires disponibles dans requirements.txt

À exécuter de préférence dans un environnement virtuel type pipenv ou pip3 :
```
pip3 install -r requirements.txt
```

Random Agent
===========
Pour tester l'agent random dans l'environnement CartPole-V1 :
```
python3 random_agent.py
```

DQLearning Agent
===========
Pour tester l'agent apprenant dans l'environnement CartPole-V1 :
```
python3 dql.py
```
- Modification des hyperparamètres dans la fonction init de la classe DQN_Agent.
- Choix de la stratégie dans la fonction init de la classe DQN_Agent, juste après les hyperparamètres (choix entre boltzmann et e-greedy).