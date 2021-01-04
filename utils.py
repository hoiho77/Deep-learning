def load_model(model_name):
  if model_name == 'alexnet' :
    from models.alexnet import Alexnet
    model = Alexnet()
    return model
  
  else : 
    print("The model_name is not exists.")
