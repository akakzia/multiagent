import numpy as np
import gym


def makeFilteredEnv(env):
  """ crate a new environment class with actions and states normalized to [-1,1] """
  acsp = env.action_space
  obsp = env.observation_space
  if not type(acsp[0])==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')
  if not type(obsp[0])==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

  env_type = type(env)

  class FilteredEnv(env_type):
    def __init__(self):
      self.__dict__.update(env.__dict__) # transfer properties

      # Observation space
      if np.any(obsp[0].high < 1e10):
        h = obsp[0].high
        l = obsp[0].low
        sc = h-l
        self.o_c = (h+l)/2.
        self.o_sc = sc / 2.
      else:
        self.o_c = np.zeros_like(obsp[0].high)
        self.o_sc = np.ones_like(obsp[0].high)

      # Action space
      h = acsp[0].high
      l = acsp[0].low
      sc = (h-l)
      self.a_c = (h+l)/2.
      self.a_sc = sc / 2.

      # Rewards
      self.r_sc = 0.1
      self.r_c = 0.

      # Special cases
      '''
      if self.spec.id == "Reacher-v1":
        print "is Reacher!!!"
        self.o_sc[6] = 40.
        self.o_sc[7] = 20.
        self.r_sc = 200.
        self.r_c = 0.
    '''
      # Check and assign transformed spaces
      self.observation_space = [gym.spaces.Box(self.filter_observation(obs.low),
                                              self.filter_observation(obs.high)) for obs in obsp]
      self.action_space = [gym.spaces.Box(-np.ones_like(ac.high),np.ones_like(ac.high)) for ac in acsp]
      def assertEqual(a,b): assert np.all(a == b), "{} != {}".format(a,b)
      #assertEqual(self.filter_action(self.action_space.low), acsp.low)
      #assertEqual(self.filter_action(self.action_space.high), acsp.high)

    def filter_observation(self,obs):
      return (obs-self.o_c) / self.o_sc

    def filter_action(self,action):
      return self.a_sc*action+self.a_c

    def filter_reward(self,reward):
      ''' has to be applied manually otherwise it makes the reward_threshold invalid '''
      return self.r_sc*reward+self.r_c

    def step(self,action):

      ac_f = np.clip(self.filter_action(action),self.action_space[0].low,self.action_space[0].high)

      obs, reward, term, info = env_type.step(self,ac_f) # super function

      obs_f = self.filter_observation(obs)

      return obs_f, reward, term, info

  fenv = FilteredEnv()

  print('True action space: ' + str(acsp[0].low) + ', ' + str(acsp[0].high))
  print('True state space: ' + str(obsp[0].low) + ', ' + str(obsp[0].high))
  print('Filtered action space: ' + str(fenv.action_space[0].low) + ', ' + str(fenv.action_space[0].high))
  print('Filtered state space: ' + str(fenv.observation_space[0].low) + ', ' + str(fenv.observation_space[0].high))

  return fenv