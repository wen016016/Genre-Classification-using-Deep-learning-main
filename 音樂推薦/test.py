import pickle

with open('history_100_epoch_tr_GRU6.pkl', 'rb') as file:
    loaded_objects = pickle.load(file)
    loaded_dicts = pickle.load(file)

# 打印列表中的对象类型
for i, obj in enumerate(loaded_objects):
    print(f"Object {i}: {type(obj)}")
for i, d in enumerate(loaded_dicts):
    print(f"Dictionary {i}: {d.keys()}")
