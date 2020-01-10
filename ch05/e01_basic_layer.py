class MultiplyLayer:

    def __init__(self):
        # forward 메소드가 호출될 때 전달되는 입력값을 저장하기 위한 변수
        # -> backward 메소드가 호출될 때 이 값들이 사용되기 때문에.
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, delta_out):
        dx = delta_out * self.y
        dy = delta_out * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx, dy = dout, dout
        return dx, dy


if __name__ == '__main__':
    apple_layer = MultiplyLayer()  # MultiplyLayer 객체 생성

    apple = 100  # 사과 한개의 가격: 100원
    n = 2  # 사과 개수: 2개
    apple_price = apple_layer.forward(apple, n)  # 순방향 전파(forward propagation)
    print('사과 2개 가격:', apple_price)

    # tax_layer를 MultiplyLayer 객체로 생성
    # tax = 1.1 설정해서 사과 2개 구매할 때 총 금액을 계산
    tax_layer = MultiplyLayer()
    tax = 1.1
    total_price = tax_layer.forward(apple_price, tax)
    print('토탈 금액:', total_price)

    # f = a * n * t
    # tax가 1 증가하면 전체 가격은 얼마가 증가? -> df/dt
    # 사과 개수가 1 증가하면 전체 가격은 얼마가 증가? -> df/dn
    # 사과 가격이 1 증가하면 전체 가격은 얼마가 증가? -> df/da

    # backward propagation(역전파)
    delta = 1.0
    dprice, dtax = tax_layer.backward(delta)
    print('dprice =', dprice)
    print('dtax =', dtax)  # df/dt: tax 변화율에 대한 전체 가격 변화율

    dapple, dn = apple_layer.backward(dprice)
    print('dapple =', dapple)  # df/da
    print('dn =', dn)  # df/dn

    # AddLayer 테스트
    add_layer = AddLayer()
    x = 100
    y = 200
    dout = 1.5
    f = add_layer.forward(x, y)
    print('f =', f)
