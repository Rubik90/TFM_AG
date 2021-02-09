import sys
import resnet50
import vgg16

if "--train_path" in sys.argv:
  train_path = sys.argv[sys.argv.index("--train_path") + 1]
else:
  print("ERROR: No value specified for parameter \"train_path\" ")
  sys.exit()

if "--val_path" in sys.argv:
  val_path = sys.argv[sys.argv.index("--val_path") + 1]
else:
  print("ERROR: No value specified for parameter \"val_path\" ")
  sys.exit()
    
if "--test_path" in sys.argv:
  test_path = sys.argv[sys.argv.index("--test_path") + 1]
else:
  print("ERROR: No value specified for parameter \"test_path\" ")
  sys.exit()
    
if "--model" in sys.argv:
  model = sys.argv[sys.argv.index("--model") + 1]
else:
  print("ERROR: No value specified for parameter \"model\" (possible values are \"resnet50\" and \"vgg_16\")")
  sys.exit()

print(model)
if model == "resnet50":
  model = resnet50.model(train_path, val_path, test_path)
elif model == "vgg16":
  model = vgg16.model(train_path, val_path, test_path)
else:
  print(str(model)+"ERROR: Invalid value for param \"model\" (the only two possible values are \"resnet50\" and \"vgg_16\")")
  sys.exit()

if __name__ == "__main__":
  model.run()
