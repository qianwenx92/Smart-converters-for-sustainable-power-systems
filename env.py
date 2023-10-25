import numpy as np
import scipy
import scipy.io
import pdb
import pypower
from network.case33 import case33
from pypower import runpf

Sbase = 100*1000000

class MAenv():
    def __init__(self, scenario_namefilePath ):
        filePath = './network/IEEE-33'
        dataPath = './data'
        Ybus = scipy.io.loadmat(f'{filePath}/Ybus.mat')
        self.Ybus = Ybus['A']
        self.n1 = self.Ybus.shape[0]
        self.v_lower = 0.95
        self.v_upper = 1.05

        self.ref = np.array([0])
        self.pv = np.array([], dtype = np.int32)
        self.pq = np.array([i  for i in range(self.n1) if (i not in self.ref) & (i not in self.pv)], dtype = np.int32)
        self.n_pq = len(self.pq)

        self._get_reference()
        self._get_load_and_gen(dataPath = dataPath)

        self.n=6
        self.obs_shape_n = [12,18,18,18,18,15]
        self.act_shape_n = [1,1,1,1,1,1]

    def getbus(self, t, wrt_reference = False, w_slack = True):
        self.P = self.P_gen[t] - self.P_l[t]
        self.Q = - self.Q_l[t]
        self.S = self.P + 1j*self.Q
        self.P_av = self.P_gen[t]
        
        if wrt_reference:
            self.S = self.S - self.S0
        
        if w_slack:
            return self.S, self.P_av[self.gen_idx], self.P
        else:
            return self.S[-self.n_pq:], self.P_av[self.gen_idx], self.P


    def reset(self):
        seed=0
        o0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        o1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        o2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        o3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        o4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        o5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        o = np.array([np.array(o0), np.array(o1), np.array(o2), np.array(o3), np.array(o4), np.array(o5)])
        return o

    def step(self, Sbus, act):
        P=Sbus.real
        Q=Sbus.imag
        i = 0
        #pdb.set_trace()
        for i, idx in enumerate(self.gen_idx):
            Q[idx] = act[i]/100;
        ppc=case33()
        ppc['bus'][:,2]=P*100
        ppc['bus'][:,3]=Q*100
        result,success=runpf.runpf(ppc)
        self.V=(result['bus'][:,7])
        reward =  - (np.abs(self.V-1).sum())/100
        agent1 = [self.Q[8]*150, self.Q[9]*150, self.Q[10]*150, self.Q[11]*150,
                    (self.V[8]-1)*20, (self.V[9]-1)*20, (self.V[10]-1)*20, (self.V[11]-1)*20,
                    self.P[8]*500, self.P[9]*500, self.P[10]*500, self.P[11]*500]
        agent2 = [self.Q[12]*150, self.Q[13]*150, self.Q[14]*150, self.Q[15]*150, self.Q[16]*150, self.Q[17]*150,
                    (self.V[12]-1)*20, (self.V[13]-1)*20, (self.V[14]-1)*20, (self.V[15]-1)*20, (self.V[16]-1)*20, (self.V[17]-1)*20,
                    self.P[12]*500, self.P[13]*500, self.P[14]*500, self.P[15]*500, self.P[16]*500, self.P[17]*500]
        agent3 = [self.Q[0]*150, self.Q[1]*150, self.Q[18]*150, self.Q[19]*150, self.Q[20]*150, self.Q[21]*150,
                    (self.V[0]-1)*20, (self.V[1]-1)*20, (self.V[18]-1)*20, (self.V[19]-1)*20, (self.V[20]-1)*20, (self.V[21]-1)*20,
                    self.P[0]*500, self.P[1]*500, self.P[18]*500, self.P[19]*500, self.P[20]*500, self.P[21]*500]
        agent4 = [self.Q[2]*150, self.Q[3]*150, self.Q[4]*150, self.Q[22]*150, self.Q[23]*150, self.Q[24]*150,
                    (self.V[2]-1)*20, (self.V[3]-1)*20, (self.V[4]-1)*20, (self.V[22]-1)*20, (self.V[23]-1)*20, (self.V[24]-1)*20,
                    self.P[2]*500, self.P[3]*500, self.P[4]*500, self.P[22]*500, self.P[23]*500, self.P[24]*500]  
        agent5 = [self.Q[5]*150, self.Q[6]*150, self.Q[7]*150, self.Q[25]*150, self.Q[26]*150, self.Q[27]*150,
                    (self.V[5]-1)*20, (self.V[6]-1)*20, (self.V[7]-1)*20, (self.V[25]-1)*20, (self.V[26]-1)*20, (self.V[27]-1)*20,
                    self.P[5]*500, self.P[6]*500, self.P[7]*500, self.P[25]*500, self.P[26]*500, self.P[27]*500]
        agent6 = [self.Q[28]*150, self.Q[29]*150, self.Q[30]*150, self.Q[31]*150, self.Q[32]*150,
                    (self.V[28]-1)*20, (self.V[29]-1)*20, (self.V[30]-1)*20, (self.V[31]-1)*20, (self.V[32]-1)*20,
                    self.P[28]*500, self.P[29]*500, self.P[30]*500, self.P[31]*500, self.P[32]*500] 
        reward_n =reward*20;
        state = np.array([np.array(agent1), np.array(agent2), np.array(agent3), np.array(agent4), np.array(agent5), np.array(agent6)])

        return state, [reward_n for i in range(6)], [False for i in range(6)],[True]


    def _get_load_and_gen(self, dataPath = './data'):
        self.load_idx = np.array([7, 8, 12, 17, 19, 25, 28, 29])-1
        load = scipy.io.loadmat(f'{dataPath}/Loads.mat')
        load = load['Loads'].transpose()
        self.P_l = np.zeros((load.shape[0], self.n1))
        for i, idx in enumerate(self.load_idx):
            self.P_l[:, idx] = load[:, i % load.shape[1]] *4 
        self.Q_l = 0 * self.P_l
        self.P_l /= Sbase;
        self.Q_l /= Sbase;

        solar_rad = scipy.io.loadmat(f'{dataPath}/Irradiance.mat')
        solar_rad = solar_rad['Irradiance'].transpose()
        self.gen_idx = np.array([10, 15, 22, 24, 27, 30])-1
        self.max_S = np.array([200, 200, 300, 300, 300, 350])*4;
        self.max_S = self.max_S * 1000 / Sbase
        Area_PV = np.array([200, 200, 300, 300, 300, 350])*4;
        PV_Irradiance_to_Power_Efficiency = 1;
        self.P_gen = np.zeros((load.shape[0], self.n1))
        gen = solar_rad * Area_PV * PV_Irradiance_to_Power_Efficiency
        gen /= Sbase
        self.P_gen[:, self.gen_idx] = gen.clip(max = self.max_S.reshape(1, -1))
