import retro

env = retro.make('SonicTheHedgehog-Genesis','GreenHillZone.Act1')

env.reset()

done = False

while not done:
    env.render()

    # action = env.action_space.sample()
    # print(action)

    action = [0,0,1,0,0,0,0,1,1,1,0,0] #ir a la derecha

    # ob -> image of the screen
    # rew -> reward
    # done -> done condition
    # info ->  diccionario de todos los valores de data

    ob, rew, done, info = env.step(action)

    print("Action ", action)
    # print("Image: ", ob.shape, "Reward: ", rew, "Done? ", done)
    # print("Info", info)
    print("Reward ", rew)

