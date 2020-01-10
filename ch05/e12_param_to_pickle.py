'''
ex11.py 에서 저장한 pickle 파일을 읽어서,
파라미터(가중치/편향 형렬)들을 화면에 출력

'''
import pickle

with open('params.pickle', mode = 'rb') as f:
    params = pickle.load(f)

for key, param in params.items():
    print(key, ':', param.shape)
