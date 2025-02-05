import retro
import numpy as np
import cv2
import neat
import pickle

# Cargar el genoma ganador
with open('winner.pkl', 'rb') as f:
    winner = pickle.load(f)

# Configuración del entorno
env = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'
)

# Tamaño de las observaciones
inx, iny, inc = env.observation_space.shape
inx = int(inx / 8)
iny = int(iny / 8)

# Crear red neuronal del ganador
net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)

# Ejecutar el nivel
ob = env.reset()
done = False
imgarray = []

while not done:
    env.render()
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))

    imgarray.clear()
    for x in ob:
        for y in x:
            imgarray.append(y)

    neuralNetOutput = net.activate(imgarray)
    ob, rew, done, info = env.step(neuralNetOutput)

env.close()
