
import os
import time

def main():
    epochs = 300
    run_iter = 10
    run_epoch = epochs // run_iter

    subj_list = ['subj1', 'subj3', 'subj4', 'subj6', 'subj7', 'subj8', 'subj9', 'subj10',
                  'subj12',  'subj17', 'subj18']

    for i in range(run_iter):
        for subj in subj_list:
            print('Run {}/{}'.format(i+1, run_iter))
            print('Subject: {}'.format(subj))
            print('Epochs: {}/{}'.format(i*run_epoch, (i+1)*run_epoch))
            print('Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            os.system('python trainAligner.py %s %d' % (subj, run_epoch))
            # print('python trainAligner.py %s %d' % (subj, run_epoch))
