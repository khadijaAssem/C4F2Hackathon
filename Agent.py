import random
import tensorflow as tf
import numpy as np
from sumolib import checkBinary  # noqa
import traci
from sumo_utils import get_total_waiting_time


class Agent:
    def __init__(self):
        self.before_action = 0
        self.step = 0
        self.tot_neg_reward = 0
        Number_States = 53
        self.Number_Actions = 12
        batch_size = 32
        Memory_Size = 64
        self._Model = Model(Number_States, self.Number_Actions, batch_size)
        with tf.Session() as SimSession:
                SimSession.run(self._Model.var_init)
        SimSession.run(self._Model.var_init)
        _memory = Memory(Memory_Size)

    #choose action
    def _choose_action(self, state):
        if random.uniform(0.0,1.0) < self._epsilongreedy:
            return random.randint(0, self._Model.Number_Actions - 3) #Random action
        else:
            return np.argmax(self._Model.predict_one(state, self.Session) ) #Use memory

    def select_action(self, state, conn=None, vehicle_ids=None):
        # 1000 sec
        self.state = state
        self.vehicle_ids = vehicle_ids
        self.conn = conn
        self.step = self.step + 1
        self.action = self._choose_action(self.state)


        # Set yellow phase if traffic signal is different from previous signal
        if self._steps != 0 and self.old_action != self.action:
            self._Set_YellowPhase(self.old_action)
            self._simulate(self._yellow_duration)
        self._Set_GreenPhaseandDuration(self.action)
        self._simulate(self._green_duration)


        self.old_action = self.action
        return self.action

    def set_episod(self, epsilongreedy):
        self._epsilongreedy = epsilongreedy

    def _learn(self):
        self.now = get_total_waiting_time (self.conn, self.vehicle_ids)

        self.reward =  self.before_action - self.now
        # Add previous state, action, reward and current state to memory
        if self._steps != 0:
            self._memory.Add_Sample((self.old_state, self.old_action, self.reward, self.state))
        self.before_action = self.now
        self.old_state = self.state

        if self.reward < 0:
            self.tot_neg_reward += self.reward

    def _replay(self):
        Batch = self._memory.Get_Samples(self._Model.batch_size)
        if len(Batch) > 0:
            states = np.array([val[0] for val in Batch])
            next_states = np.array([val[3] for val in Batch])
            QSA = self._Model.predict_batch(states, self.Session)
            QSATarget = self._Model.predict_batch(next_states, self.Session)
            x = np.zeros((len(Batch), self._Model.Number_States))
            y = np.zeros((len(Batch), self._Model.Number_Actions))
            for i, b in enumerate(Batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3]
                Current_Q = QSA[i]
                Current_Q[action] = reward + self._gamma * np.amax(QSATarget[i])
                x[i] = state
                y[i] = Current_Q
            self._Model.train_batch(self.Session, x, y)


# Reinforcement learning model
class Model:
    def __init__(self, Number_States, Number_Actions, batch_size):
        self._Number_States = Number_States
        self._Number_Actions = Number_Actions
        self._batch_size = batch_size

        self._states = None
        self._actions = None

        self._logits = None
        self._optimizer = None
        self._var_init = None

        self._define_model()

    # Create neural network
    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._Number_States], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._Number_Actions], dtype=tf.float32)

        fc1 = tf.layers.dense(self._states, 33, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 33, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 33, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc3, self._Number_Actions)

        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, SimSession):
        return SimSession.run(self._logits, feed_dict={self._states: state.reshape(1, self.Number_States)})

    def predict_batch(self, states, SimSession):
        return SimSession.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, SimSession, x_batch, y_batch):
        SimSession.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    @property
    def Number_States(self):
        return self._Number_States

    @property
    def Number_Actions(self):
        return self._Number_Actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init


# Class for storying and receiving memory
class Memory:
    def __init__(self, Memory_Size):
        self._Memory_Size = Memory_Size
        self.Samples = []

    def Get_Samples(self, Number_Samples):
        if Number_Samples > len(self.Samples):
            return random.sample(self.Samples, len(self.Samples))
        else:
            return random.sample(self.Samples, Number_Samples)

    def Add_Sample(self, sample):
        self.Samples.append(sample)
        if len(self.Samples) > self._Memory_Size:
            self.Samples.pop(0)

# if __name__ == "__main__":
#
#     gui = True
#     total_episodes = 100
#     gamma = 0.75
#     batch_size = 32
#     Memory_Size = 3200
#     path = "./model/model_1g/"
#     # ----------------------
#
#     Number_States = 53
#     Number_Actions = 12
#     Max_Steps = 3600
#     Green_Duration = 10
#     Yellow_Duration = 7
#
#     # Change to False if Simulation GUI must be shown
#     if gui == True:
#         sumoBinary = checkBinary('sumo')
#     else:
#         sumoBinary = checkBinary('sumo-gui')
#
#     model = Model(Number_States, Number_Actions, batch_size)
#     memory = Memory(Memory_Size)
#
#     with tf.Session() as SimSession:
#         SimSession.run(model.var_init)
#         sim_runner = RunSimulation(SimSession, model, memory, traffic_gen, total_episodes, gamma, Max_Steps,
#                                    Green_Duration, Yellow_Duration, SUMO_Command)
#         episode = 0
#
#     traci.start([sumoBinary, "-c", "data/cross.sumocfg",
#                  "--time-to-teleport", "-1",
#                  "--tripinfo-output", "tripinfo.xml", '--start', '-Q'], label='contestant')
#     # Connection to simulation environment
#     conn = traci.getConnection("contestant")
