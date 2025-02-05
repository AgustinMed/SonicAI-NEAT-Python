import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make('SonicTheHedgehog2-Genesis','EmeraldHillZone.Act1')

imgarray = []

xpos_end = 0

def evaluate(genomes, config):

    for genome_id, genome in genomes:

        ob = env.reset() # Variable de observación
        ac = env.action_space.sample() # Variable de acción

        inx, iny, inc = env.observation_space.shape # size of the image created by the emulator (resolution), x, y and colors
        # inx, iny = 32, 32  # Tamaño reducido de la imagen

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config) #network for generete the inputs in the emulator

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        # cv2.namedWindow('main', cv2.WINDOW_NORMAL)

        while not done: #while sonic is still alive
            env.render()
            frame += 1
            # scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            # scaledimg = cv2.resize(scaledimg, (iny,inx))
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            # cv2.imshow('main', scaledimg)
            # cv2.waitKey(1)
            for x in ob: #Compress the 2D image to a single array of values
                for y in x:
                    imgarray.append(y) #put the value of each pixel to a list

            neuralNetOutput = net.activate(imgarray)


            # print(len(imgarray), neuralNetOutput)
            ob, rew, done, info = env.step(neuralNetOutput)

            imgarray.clear()
            
            xpos = info['x'] #Sonic x position

            xpos_end = info['screen_x_end']

            if xpos > xpos_max:
                fitness_current += 1 # Cada vez que sonic va un poco más a la derecha obtiene 1 punto
                xpos_max = xpos

            # if xpos == xpos_end and xpos > 500:
            #     fitness_current += 100000
            #     done = True

            if xpos >= xpos_end and xpos > 500:
                fitness_current += 100000
                done = True


            # fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current

            # print(neuralNetOutput)


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

# p = neat.Population(config)

# p.add_reporter(neat.StdOutReporter(True))
# stats = neat.StatisticsReporter()
# p.add_reporter(stats)
# p.add_reporter(neat.Checkpointer(5))

# winner = p.run(evaluate)

# with open('winner.pkl', 'wb') as output:
#     pickle.dump(winner, output, 1)

# Cargar el checkpoint si existe
checkpoint_file = 'neat-checkpoint-8'
p = neat.Population(config)

# Intentamos cargar el checkpoint
try:
    checkpoint = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    p = checkpoint  # Usamos el objeto restaurado
    print(f"Checkpoint {checkpoint_file} cargado con éxito.")
except Exception as e:
    print(f"No se pudo cargar el checkpoint {checkpoint_file}: {e}")

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

# Continúa el entrenamiento o evalúa
winner = p.run(evaluate)

# Guardar el modelo ganador
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)