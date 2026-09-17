from pathlib import Path
import tensorflow as tf

from hsdetect import Engine


IS_ENGINE_MODULE = True

def build_dl_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'DLEngine':
        return DLEngine(**kwargs)
    else:
        raise TypeError(f'Unknown Inference Engine {name}!')


import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class DLEngine(Engine):
    
    def __init__(self,
             SensorType:int,
             **attr_dict:dict):
        
        self.SensorType = SensorType  
        self.dl_model = attr_dict.pop('dl_model',None)
        
    @property
    def SensorType(self):
        return self._SensorType
    
    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType
        
    @property
    def dl_model(self):
        return self._dl_model
    
    @dl_model.setter
    def dl_model(self,dl_model):
        if dl_model is None:
            self._dl_model = None
        elif isinstance(dl_model, Path):
            self._dl_model = tf.keras.models.load_model(dl_model)
        elif isinstance(dl_model,tf.keras.Model):
            self._dl_model = dl_model
        else:
            raise Exception(f'Model {dl_model} not recognized.')


    def save(self):
        """
        Overrides standard save method, which only stores attributes, but 
        is not capable of storing objects which have a save() method of 
        their own.

        Returns
        -------
        None.

        """
        
        attr_dict = {}
        
        for attr_name in dir(self):
            # get the attribute *on the class*, not the instance
            attr = getattr(self.__class__, attr_name, None)
        
            # check if it's a property
            if isinstance(attr, property):
                
                # Retrieve the value of the property attr
                attr_val = getattr(self, attr_name)
                
                # If the property has a save() method, save the return value of
                # that method
                if hasattr(attr_val, "save") and callable(attr_val.save):
                    attr_dict[attr_name] = attr_val.save()
                else:
                    attr_dict[attr_name] = attr_val
                
        # Append name
        attr_dict['name'] = self.name
                
        return attr_dict