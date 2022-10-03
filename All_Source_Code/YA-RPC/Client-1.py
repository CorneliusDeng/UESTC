import json, time, struct, socket



def rpc(sock, in_, params, args=None):
    # 将请求序列化打包
    request = json.dumps({"in": in_, "params": params, 'args':args})
    length_prefix =  struct.pack('I', len(request))
    # 发送请求和请求体
    sock.sendall(length_prefix)
    sock.sendall(str.encode(request))
    print('send data:', str.encode(request))

    # 等待接受响应
    try:
        #设置响应时间 以实现At-least-once语义
        sock.settimeout(60)
        # 接受响应并且得到响体
        length_prefix = sock.recv(4)
        length, = struct.unpack('I', length_prefix)
        body = sock.recv(length)        # 响应消息体
        print('recv:', body)
        response = json.loads(body)
        return response['out'], response['result'] # 返回响应类型和结果
    except Exception as e:
        # 响应时间超过时、重新发送请求
        print(e)
        return rpc(sock, in_, params, args)

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 8080))

    # # 测试sum函数结果
    # for i in range(2):
    #     out, result = rpc(s, 'sum', '1,2', 'int')
    #     out, result = rpc(s, 'sum', '1,2.4', 'float')
    #
    #     time.sleep(5)

    # # 测试uppercase函数结果
    # print('测试Uppercase')
    # for i in range(2):
    #     out, result = rpc(s, 'uppercase', 'iloveyou', 'str')
    #     out, result = rpc(s, 'uppercase', 'Apple', 'str')
    #     print(out, result)
    #     time.sleep(5)

    # # 测试At-least-once语义
    # print('测试At-least-once语义')
    # out, result = rpc(s, 'uppercase', 'iloveyou', 'str')
    # print(out, result)

    # 测试基本类型格式
    # print('测试基本类型格式')
    # out, result = rpc(s, 'sum', '1,2', 'int')
    # out, result = rpc(s, 'sum', '1,2.4', 'float')
    # out, result = rpc(s, 'sum', 'a,b', 'float')

    # 测试并发请求
    for i in range(2):
        out, result = rpc(s, 'sum', '1,2', 'int')
        time.sleep(1)
        out, result = rpc(s, 'sum', '1,2.4', 'float')
        time.sleep(1)
        out, result = rpc(s, 'sum', '1,2', 'int')
        time.sleep(1)
        out, result = rpc(s, 'sum', '1,2.4', 'float')
        time.sleep(1)
    s.close()

