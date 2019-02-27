from network import Network
from regularization import Regularization
import numpy as np


class SimpleRegularizerFunction:
    
    def __init__(self,max_iterations,min_e):
        self.max_iterations=max_iterations
        self.min_e=min_e
    
    def propagation(self,net):
        
        iteration = 0
        dim = -1
        
        while True:
        
            nodes = net.getNodes()
            for node in nodes:
                if(net.getNodeData(node,'labeled')==True): continue
                f = net.getNodeData(node,'f')
                dim = len(f)
                f_temp = np.array([0]*dim)
                neighbors = net.getNeighbors(node)

                normalization = 0
                for neighbor in neighbors:
                    penalty = net.getDegree(neighbor)
                    if(net.getNodeData(neighbor,'labeled')==True): penalty=1
                    normalization += 1/penalty
                    f_neighbor = net.getNodeData(neighbor,'f')
                    f_temp = f_temp + (f_neighbor/penalty)

                f_temp = f_temp/normalization
                net.setNodeData(node,'f_temp',f_temp)

            e = 0
            for node in nodes:
                if(net.getNodeData(node,'labeled')==True): continue
                f = net.getNodeData(node,'f')
                f_temp = net.getNodeData(node,'f_temp')
                e += np.sum(np.square(f - f_temp))
                net.setNodeData(node,'f',f_temp)
                f = None
        
            iteration += 1
            print("iteration="+str(iteration)+"\te="+str(e))
            
            if iteration >= self.max_iterations: break
            if e < self.min_e: break
        
        
        for node in nodes:
            if(net.getNodeData(node,'labeled')==True):
                f_temp = np.array([0]*dim)
                neighbors = net.getNeighbors(node)
                normalization = 0
                for neighbor in neighbors:
                    penalty = net.getDegree(neighbor)
                    if(net.getNodeData(neighbor,'labeled')==True): penalty=1
                    normalization += 1/penalty
                    f_neighbor = net.getNodeData(neighbor,'f')
                    f_temp = f_temp + (f_neighbor/penalty)
                f_temp = f_temp/normalization
                net.setNodeData(node,'f_temp',f_temp)

        for node in nodes:
            if(net.getNodeData(node,'labeled')==True):
                net.setNodeData(node,'f',net.getNodeData(node,'f_temp'))
                
                
net = Network()
net.setMemoryCache(True)
net.loadCSV('network-acm.csv')

srf = SimpleRegularizerFunction(30,0.05)
reg = Regularization(net,'labels-acm.csv')
reg.propagate(srf.propagation)

net.printNodes()
