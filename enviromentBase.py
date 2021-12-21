class EnviromentBase:

    def __init__(self):
        pass
    
    def makeAction(self, action):
        # input action 
        # output reward
        raise NotImplementedError
    
    def newEpisode(self):
        # get new episode
        raise NotImplementedError

    def getState(self):
        raise NotImplementedError
