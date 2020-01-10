'''
교재 p.160 그림 5-15의 빈칸 채우기.
apple = 100원, n_a = 2개
orange = 150원, n_o = 3개
tax = 1.1
라고 할 때,

전체 과일 구매 금액을 AddLayer와 MultiplyLayer를 사용해서 계산하세요.
df/dapple, df/dn_a, df/dorange, df/dn_o, df/dtax 값들도 각각 계산하세요.

'''
from ch05.e01_basic_layer import MultiplyLayer, AddLayer

apple = 100
n_a = 2
orange = 150
n_o = 3
tax = 1.1

mul_gate_a = MultiplyLayer()
q_a = mul_gate_a.forward(apple, n_a)
mul_gate_o = MultiplyLayer()
q_o = mul_gate_o.forward(orange, n_o)

add_gate = AddLayer()
p = add_gate.forward(q_a, q_o)

mul_gate_p = MultiplyLayer()
f = mul_gate_p.forward(p, tax)

print('f =', f)
d_p, d_tax = mul_gate_p.backward(1)
d_q_a, d_q_o = add_gate.backward(d_p)
d_apple, d_n_a = mul_gate_a.backward(d_q_a)
d_orange, d_n_o = mul_gate_o.backward(d_q_o)

print('df/d_apple = ', d_apple)
print('df/dn_a =', d_n_a)
print('df/d_orange = ', d_orange)
print('df/dn_o =', d_n_o)
print('df/dtax =', d_tax)




