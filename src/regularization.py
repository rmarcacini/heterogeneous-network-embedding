import random
import numpy as np

class Regularization:
    
    
    def __init__(self,net,labeled_data):
        self.net=net
        counter=1
        dim = -1
        with open(labeled_data) as myfile:
            for line in myfile:  
                data = line.split('\t')
                if(len(data)!=2):
                    print("Warning: malformed data in line "+str(counter)+" ["+line+"]")
                    continue

                ln = data[0].split(':')
                node = data[0].strip()
                y=np.array((list(map(float,data[1].strip().split(',')))))
                
                dim=(len(y))
                
                self.net.setNodeData(node,'labeled',True)
                self.net.setNodeData(node,'y',y)
                
                counter += 1
        
        
        for node in self.net.getNodes():
            if(self.net.getNodeData(node,'labeled')==None):
                self.net.setNodeData(node,'labeled',False)
                self.net.setNodeData(node,'f',np.array([0]*dim))

        for node in self.net.getNodes():
            if(self.net.getNodeData(node,'labeled')==True):
                self.net.setNodeData(node,'f',self.net.getNodeData(node,'y'))
                
            
        

    def propagate(self,regfunction):
        regfunction(self.net)
        
