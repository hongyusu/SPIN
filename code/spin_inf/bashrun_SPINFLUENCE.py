
import os
import sys
import commands
sys.path.append('/home/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)

def singlerun(filename,c,g,nb,penalty,losstype,node):
        logging.info(' --> (f)%s,(C)%s,(g)%s,(nb)%s,(p)%s,%s,%s' %( filename,c,g,nb,penalty,losstype,node))
        try:
                with open("../results/T_%s_%s_%s_%s_%s_%s_SPINFLUENCE.mat" % (filename,c,g,nb,penalty,losstype)): pass
        except:
                #print "T_%s_%s_%s_%s_%s_%s_SPINFLUENCE.mat" % (filename,c,g,nb,penalty,losstype)
                os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /home/group/urenzyme/workspace/forecml/spinfluence/; nohup matlab -nodisplay -nosplash -nodesktop -r "run_SPINFLUENCE '%s' '%s' '%s' '%s' '%s' '%s' " > /var/tmp/tmp_%s_%s_%s_%s_%s_%s_SPINFLUENCE' """ % (node,c,g,nb,penalty,losstype,filename,filename,c,g,nb,penalty,losstype) )
        logging.info(' --| (f)%s,(C)%s,(g)%s,(nb)%s,(p)%s,%s,%s' %( filename,c,g,nb,penalty,losstype,node))
        time.sleep(10)
        pass

def run():
        cluster = get_free_nodes()[0]
        jobs=[]
        n=0

        filenames=['memeS','memeM','memeL']
        for year in [2000]:
                for copaper in [5,10,15]:
                        for v in [2000,1000,100,300,500,700]:
                                filenames.append('%d_%d_%d' % (year,copaper,v))
        for filename in filenames:
                c='100'
                g='0.8'
                for nb in ['1','3','5','7']:
                        for losstype in ['exp']:
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
                                        time.sleep(5)
                        time.sleep(10)
        for job in jobs:
                job.join()
                                                                                

run()

                                        



