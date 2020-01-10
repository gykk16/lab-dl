'''
Perceptron(퍼셉트론): 다수의 신호를 입력받아서, 하나의 신호를 출력

   AND : 두 입력이 모두 1일때 출력이 1, 그 외에는 0
   NAND : AND 출력의 반대 (NOT)
   OR : 두 입력 중 적어도 하나가 1이면 출력이 1, 그 외에는 0
   XOR : 두 입력 중 하나는 1, 다른 하나는 0일때만 출력이 1 그 외에는 0

'''


def and_gate(x1, x2):
    w1, w2 = 1, 1  # 가중치(weight)
    b = -1  # 편향(bias)
    y = x1 * w1 + x2 * w2 + b
    if y > 0:
        return 1
    else:
        return 0


def nand_gate(x1, x2):
    # if and_gate(x1, x2) > 0:
    #     return 0
    # else:
    #     return 1

    w1, w2 = 0.5, 0.5  # 가중치(weight)
    b = 0  # 편향(bias)
    y = x1 * w1 + x2 * w2 + b
    if y < 1:
        return 1
    else:
        return 0


def or_gate(x1, x2):
    w1, w2 = 0.5, 0.5
    b = 0.5
    y = x1 * w1 + x2 * w2 + b
    if y >= 1:
        return 1
    else:
        return 0


def xor_gate(x1, x2):
    ''' XOR(Exclusive OR: 배타적 OR)
        신형 관계씩 (y = x1 * w1 + x2 * w2 + b) 하나만 이용해서는 만들 수 없다
        NAND, OR, AND 조합해랴 가능
        '''

    z1 = nand_gate(x1, x2)
    z2 = or_gate(x1, x2)

    return and_gate(z1, z2)


if __name__ == '__main__':
    # AND GATE
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'AND({x1}, {x2}) -> {and_gate(x1, x2)}')

    # NAND GATE
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'NAND({x1}, {x2}) -> {nand_gate(x1, x2)}')

    # OR GATE
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'OR({x1}, {x2}) -> {or_gate(x1, x2)}')

    # XOR GATE
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'XOR({x1}, {x2}) -> {xor_gate(x1, x2)}')

