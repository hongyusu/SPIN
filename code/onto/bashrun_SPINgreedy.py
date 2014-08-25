
import os
import sys
import commands
sys.path.append('/cs/taatto/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)

def singlerun(filename,c,g,nb,penalty,losstype,node):
        try:
                with open("../results/T_%s_%s_%s_%s_%s_%s_SPINgreedy.mat" % (filename,c,g,nb,penalty,losstype)): pass
        except:
                logging.info('\t--> (f)%s,(C)%s,(g)%s,(nb)%s,(p)%s,%s,%s' %( filename,c,g,nb,penalty,losstype,node))
                #print "T_%s_%s_%s_%s_%s_%s_SPINgreedy.mat" % (filename,c,g,nb,penalty,losstype)
                os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /cs/taatto/group/urenzyme/workspace/structured_prediction_network_response/spin_gdy/; nohup matlab -nodisplay -nosplash -nodesktop -r "run_SPINgreedy '%s' '%s' '%s' '%s' '%s' '%s' " > /var/tmp/tmp_%s_%s_%s_%s_%s_%s_SPINgreedy' """ % (node,c,g,nb,penalty,losstype,filename,filename,c,g,nb,penalty,losstype) )
                logging.info('\t--| (f)%s,(C)%s,(g)%s,(nb)%s,(p)%s,%s,%s' %( filename,c,g,nb,penalty,losstype,node))
        time.sleep(5)
        pass

def run():
        cluster = get_free_nodes()[0]
        jobs=[]
        n=0

        filenames=['memeS','memeM','memeL']
        filenames=[]
        for year in [2000]:
                for copaper in [5,10,15]:
                        for v in [100,300,500,700,1000,2000]:
                                filenames.append('%d_%d_%d' % (year,copaper,v))
        for filename in filenames:
                c='100'
                g='0.8'
                for nb in ['1','3','5','7']:
                        for losstype in ['dif','exp']:
                                if losstype == 'exp':
                                        penaltyrange = ['0.3','0.5','0.7','1','3','5','7']
                                else:
                                        penaltyrange = ['0.1','0.5','0.8']
                                for penalty in penaltyrange:
                                        node = cluster[n%len(cluster)]
                                        n+=1
                                        p=multiprocessing.Process(target=singlerun, args=(filename,c,g,nb,penalty,losstype,node,))
                                        jobs.append(p)
                                        p.start()
                                        time.sleep(15)
                        time.sleep(60)
                time.sleep(300)
        for job in jobs:
                job.join()
                                                                                

run()

                                        



