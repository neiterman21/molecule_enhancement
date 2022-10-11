import wandb
import os

from typing import Dict, Any

MODEL_NAME = "model.pt"


class StraitjakcetDetached:
    def __init__(self, projectName, userName, id):

        self.run = wandb.init(project=projectName, resume=False, id=id)
        self.runDir = wandb.run.dir
        self._userName = userName

        self._api = wandb.Api()
        self._runApi = self._api.run(self.run.path)

    def Save(self, filename):
        wandb.save(filename)

    def Restore(self, filename, replace=False):
        wandb.restore(filename, self.run.path, replace=replace)

    def Log(self, logDict, upload = True):
        wandb.log(logDict, commit=upload)


class Straitjakcet(StraitjakcetDetached):
    """
        wandb wrapper because they keep changing stuff around.
        ANY interaction with their API is confined in this comfortable-to-debug-when-it-breaks straitjacket.

    """

    def __init__(self, projectName, userName, id, entity = None, resume=False):
       # os.environ["WANDB_API_KEY"] = "e67eaf73adb041e2f73f4d41c038f40143ff8124"
        if not id is None: 
            self.run = wandb.init(project=projectName, resume="allow", entity=entity, id=id)
        else:
            self.run = wandb.init(project=projectName, resume="allow", entity=entity)
        self.runDir = wandb.run.dir
        self.modelDir = os.path.join(self.runDir, MODEL_NAME)
        self._userName = userName

        if resume and self.run.resumed:
            self.RestoreModel()

        self._api = wandb.Api()
        self._runApi = self._api.run(self.run.path)

        #code = wandb.Artifact('project-source', type='code')
        #code.add_file('codes/models/archs/CEL.py')
        #code.add_file('codes/data/CEL_ds.py')
        #wandb.run.use_artifact(code)

        #self.Save('codes/models/archs/CEL.py')
        #self.Save('codes/data/CEL_ds.py')

    def RestoreModel(self):
        self.Restore(MODEL_NAME, replace=True)

    def SaveModel(self):
        self.Save(MODEL_NAME)

    def WatchModel(self, model):
        wandb.watch(model)

class wandb_logger:
    def __init__(self,opt):
        self.opt = opt
        self.sj = Straitjakcet(opt['project'] , opt['user'],opt['id'],opt['entity'])
    
    def log(self,msg : dict):
        self.sj.Log(msg)


