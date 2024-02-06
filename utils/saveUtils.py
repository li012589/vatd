import subprocess, os

def createWorkSpace(path):
    savingPath = os.path.join(path, "savings")
    recordPath = os.path.join(path, "records")
    picPath = os.path.join(path, "pic")

    if not os.path.exists(savingPath):
        os.makedirs(savingPath)
    if not os.path.exists(recordPath):
        os.makedirs(recordPath)
    if not os.path.exists(picPath):
        os.makedirs(picPath)


def cleanSaving(path,epoch,keptEpoch,name):
    if epoch >= keptEpoch:
        toRemove = os.path.join(path, "savings", name+"_epoch_"+str(epoch-keptEpoch)+".saving")
        if os.path.exists(toRemove):
            os.remove(toRemove)
        toRemove = os.path.join(path, "savings", name+"_epoch_"+str(epoch-keptEpoch)+"_opt.saving")
        if os.path.exists(toRemove):
            os.remove(toRemove)

        # cmd =["rm","-rf",path+"records/"+name+"Record_epoch"+str(epoch-keptEpoch)+".hdf5"]
        # subprocess.check_call(cmd)