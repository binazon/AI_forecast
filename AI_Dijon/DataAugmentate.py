from typing import List
#######################################METHODS#####################################################################
#augmenting the origin nbDi datas
#getting frequence of nbDi in nbDi column
def freq_nbDi(data) -> List:
    tab = []
    pos = 1
    for i in data[:,1]:
        tab.append(data[:,1][0:pos].tolist().count(i))
        pos = pos + 1
    return tab
#getting all peak all nbDi
def is_peak_nbDi(data) -> List:
    peak = []
    for i in data[:,1]:
        if(int(i) >= 6):
            peak.append(1)
        else:
            peak.append(0)
    return peak
#getting all ways without request of intervention
def is_request_nbDi(data) -> List : 
    request = []
    for i in data[:,1]:
        if(int(i) > 0):
            request.append(1)
        else:
            request.append(0)
    return request